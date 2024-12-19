import os
import torch
import numpy as np
from datetime import datetime
import time
import sys
from omegaconf import OmegaConf
import importlib

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter

from dataset import load_data
from tools import (
    get_model_parameters_number,
    train_val_one_epoch,
    eval_model,
    export_onnx,
)

from args_train import args

from setup_logger import setup_logger


if args.histos:
    from sig_bkg_eval import plot_sig_bkg_distributions, plot_roc_curve
if args.history:
    from plot_history import read_from_txt, plot_history


if __name__ == "__main__":
    start_time = time.time()

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # main_dir = f"out/{timestamp}_{args.name}"
    main_dir = args.output_dir
    name = main_dir.split("/")[-1]

    best_vloss = 1_000_000.0
    best_vaccuracy = 0.0
    best_epoch = -1
    best_model_name = ""

    loaded_epoch = -1

    cfg = OmegaConf.load(args.config)

    n_epochs = args.epochs if args.epochs else cfg.epochs

    if not cfg.learning_rate_schedule in ["constant", "linear"]:
        raise ValueError("learning_rate_schedule must be either 'constant' or 'linear'")
    assert cfg.learning_rate > 0, "learning_rate must be positive"

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    ML_model = importlib.import_module(cfg.ML_model)

    if args.load_model or args.eval_model:
        main_dir = os.path.dirname(
            args.load_model if args.load_model else args.eval_model
        ).replace("models", "")
    else:
        try:
            os.makedirs(main_dir)
        except FileExistsError:
            # ask the user if they want to overwrite the directory
            print(f"Directory {main_dir} already exists")
            if args.overwrite:
                print("Overwriting...")
                os.system(f"rm -rf {main_dir}")
                os.makedirs(main_dir)
            else:
                print("Do you want to overwrite it? (y/n)")
                answer = input()
                if answer == "y":
                    os.system(f"rm -rf {main_dir}")
                    os.makedirs(main_dir)
                else:
                    print("Exiting...")
                    sys.exit(0)
    # writer = SummaryWriter(f"runs/DNN_trainer_{timestamp}")
    # Create the logger
    logger = setup_logger(f"{main_dir}/logger_{name}.log")

    logger.info("cfg:\n - %s", "\n - ".join(str(it) for it in cfg.items()))

    logger.info("args:\n - %s", "\n - ".join(str(it) for it in args.__dict__.items()))

    input_variables = cfg.input_variables
    signal_list = cfg.signal_list
    background_list = cfg.background_list

    # Load data
    (
        training_loader,
        val_loader,
        test_loader,
        train_size,
        val_size,
        test_size,
        X_fts,
        X_lbl,
        batch_size,
    ) = load_data(args, cfg)
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]
        device = torch.device(gpus[0])
    else:
        gpus = None
        device = torch.device("cpu")
    logger.info(f"Using {device} device")

    input_size = X_fts.size(1) - 1

    # Get model
    model, loss_fn, optimizer, scheduler = ML_model.get_model(
        input_size, device, cfg.learning_rate, cfg.learning_rate_schedule, n_epochs
    )
    num_parameters = get_model_parameters_number(model)

    logger.info(f"Number of parameters: {num_parameters}")

    if gpus is not None and len(gpus) > 1:
        # model becomes `torch.nn.DataParallel` w/ model.module being the original `torch.nn.Module`
        model = torch.nn.DataParallel(model, device_ids=gpus)

    if args.load_model or args.eval_model:
        checkpoint = torch.load(args.load_model if args.load_model else args.eval_model)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loaded_epoch = checkpoint["epoch"]
        best_model_name = args.load_model if args.load_model else args.eval_model
        with open(f"{main_dir}/logger_{name}.log", "r") as f:
            for line in reversed(f.readlines()):
                if "Best epoch" in line:
                    # get line from Best epoch onwards
                    line = line.split("Best epoch")[1]
                    best_epoch = int(line.split(",")[0].split("#")[1])
                    best_vloss = float(line.split(",")[1].split(":")[1])
                    best_vaccuracy = float(line.split(",")[2].split(":")[1])
                    break
        logger.info(
            f"Loaded model from %s, epoch %d, best val loss: %.4f, best val accuracy: %.4f"
            % (best_model_name, loaded_epoch, best_vloss, best_vaccuracy)
        )

    if not args.eval_model:
        for epoch in range(n_epochs):
            if epoch <= loaded_epoch:
                continue
            time_epoch = time.time()
            # Turn on gradients for training
            print("\n\n\n")

            # train
            avg_loss, avg_accuracy, *_ = train_val_one_epoch(
                True,
                epoch,
                model,
                training_loader,
                loss_fn,
                optimizer,
                device,
                time_epoch,
                scheduler,
            )

            logger.info("time elapsed: {:.2f}s".format(time.time() - time_epoch))

            # validate
            (
                avg_vloss,
                avg_vaccuracy,
                best_vloss,
                best_vaccuracy,
                best_epoch,
                best_model_name,
            ) = train_val_one_epoch(
                False,
                epoch,
                model,
                val_loader,
                loss_fn,
                optimizer,
                device,
                time_epoch,
                None,
                main_dir,
                best_vloss,
                best_vaccuracy,
                best_epoch,
                best_model_name,
            )

            logger.info(
                "EPOCH # %d: loss train %.4f,  val %.4f" % (epoch, avg_loss, avg_vloss)
            )
            logger.info(
                "EPOCH # %d: acc train %.4f,  val %.4f"
                % (epoch, avg_accuracy, avg_vaccuracy)
            )
            logger.info(
                "Best epoch # %d, best val loss: %.4f, best val accuracy: %.4f"
                % (best_epoch, best_vloss, best_vaccuracy)
            )

            # Log the running loss averaged per batch
            # for both training and validation

            # writer.add_scalars(
            #     "Training vs. Validation Loss",
            #     {"Training": avg_loss, "Validation": avg_vloss},
            #     epoch,
            # )
            # writer.add_scalars(
            #     "Training vs. Validation Accuracy",
            #     {"Training": avg_accuracy, "Validation": avg_vaccuracy},
            #     epoch,
            # )

            # writer.flush()
            epoch += 1
            logger.info("time elapsed: {:.2f}s".format(time.time() - time_epoch))

        if args.history:
            # plot the training and validation loss and accuracy
            print("\n\n\n")
            logger.info("Plotting training and validation loss and accuracy")
            train_accuracy, train_loss, val_accuracy, val_loss = read_from_txt(
                f"{main_dir}/logger_{name}.log"
            )

            plot_history(
                train_accuracy,
                train_loss,
                val_accuracy,
                val_loss,
                main_dir,
                False,
            )
    if args.onnx:
        # export the model to ONNX
        print("\n\n\n")
        logger.info("Exporting model to ONNX")
        model.train(False)
        # move model to cpu
        model.to("cpu")
        export_onnx(
            model,
            best_model_name if not args.eval_model else args.eval_model,
            batch_size,
            input_size,
            "cpu",
        )

    if args.eval or args.eval_model:
        # evaluate model on test_dataset loadining the best model
        print("\n\n\n")
        logger.info("Evaluating best model on test and train dataset")
        print("================================")

        # load best model
        model.load_state_dict(
            torch.load(best_model_name if not args.eval_model else args.eval_model)[
                "state_dict"
            ]
        )
        model.train(False)
        model.to(device)

        eval_epoch = loaded_epoch if args.eval_model else best_epoch
        logger.info("Training dataset\n")
        score_lbl_array_train, loss_eval_train, accuracy_eval_train = eval_model(
            model,
            training_loader,
            loss_fn,
            "training",
            device,
            eval_epoch,
        )

        logger.info("\nTest dataset")
        score_lbl_array_test, loss_eval_test, accuracy_eval_test = eval_model(
            model,
            test_loader,
            loss_fn,
            "test",
            device,
            eval_epoch,
        )
        print("================================")
        logger.info(
            "Best epoch # %d, loss val: %.4f, accuracy val: %.4f"
            % (best_epoch, best_vloss, best_vaccuracy)
        )
        logger.info(
            "Eval epoch # %d,  loss train: %.4f,  loss test: %.4f"
            % (eval_epoch, loss_eval_train, loss_eval_test)
        )
        logger.info(
            "Eval epoch # %d,  acc train: %.4f,  acc test: %.4f"
            % (eval_epoch, accuracy_eval_train, accuracy_eval_test)
        )
        train_test_fractions = np.array([cfg.train_fraction, cfg.test_fraction])

        # save the score and label arrays
        np.savez(
            f"{main_dir}/score_lbl_array.npz",
            score_lbl_array_train=score_lbl_array_train,
            score_lbl_array_test=score_lbl_array_test,
            train_test_fractions=train_test_fractions,
        )

        # plot the signal and background distributions
        if args.histos:
            print("\n\n\n")
            logger.info("Plotting signal and background distributions")
            plot_sig_bkg_distributions(
                score_lbl_array_train,
                score_lbl_array_test,
                main_dir,
                False,
                [0.3363, 0.3937], #TODO:remove default values
                train_test_fractions[1],
            )
        if args.roc:
            print("\n\n\n")
            logger.info("Plotting ROC curve")
            plot_roc_curve(score_lbl_array_test, main_dir, False)

    logger.info("Saved output in %s" % main_dir)

    logger.info("Total time: %.1f" % (time.time() - start_time))
