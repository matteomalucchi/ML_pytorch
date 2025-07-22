import os
import torch
import numpy as np
import datetime
import time
import sys
from omegaconf import OmegaConf
import importlib

# PyTorch TensorBoard support
# from torch.utils.tensorboard import SummaryWriter

from ml_pytorch.utils.dataset import load_data
from ml_pytorch.utils.tools import (
    get_model_parameters_number,
    train_val_one_epoch,
    eval_model,
    export_onnx,
    create_DNN_columns_list,
    save_pytorch_model,
)

from ml_pytorch.utils.args_train import args

from ml_pytorch.utils.setup_logger import setup_logger

from ml_pytorch.utils.early_stopper import EarlyStopper


def main():
    start_time = time.time()
    file_dir = os.path.dirname(__file__)
    default_cfg = OmegaConf.load(f"{file_dir}/../defaults/default_configs.yml")
    cfg = default_cfg

    if args.load_model and not args.config:
        cfg.output_dir = os.path.dirname(args.load_model).replace(
            os.path.dirname(args.load_model).split("/")[-1], ""
        )
        # find the yml file
        cfg_file_name = f"{cfg.output_dir}/config_parameters.yml"
        cfg_file = OmegaConf.load(cfg_file_name)
    elif args.eval_model and not args.config:
        cfg.output_dir = os.path.dirname(args.eval_model).replace(
            os.path.dirname(args.eval_model).split("/")[-1], ""
        )
        # find the yml file
        cfg_file_name = f"{cfg.output_dir}/config_parameters.yml"
        cfg_file = OmegaConf.load(cfg_file_name)
    else:
        if not args.config:
            raise ValueError("Choose the config")
        cfg_file_name = args.config
        cfg_file = OmegaConf.load(cfg_file_name)

    for key, val in cfg_file.items():
        cfg[key] = val

    for key, val in args.__dict__.items():
        if val is not None:
            cfg[key] = val

    if cfg.histos:
        from ml_pytorch.scripts.sig_bkg_eval import (
            plot_sig_bkg_distributions,
            plot_roc_curve,
        )
    if cfg.history:
        from ml_pytorch.scripts.plot_history import read_from_txt, plot_history, plot_lr

    #base dir is /work/<username>/
    base_dir = f"/work/{os.environ['USER']}"
    # check if exists
    if not os.path.exists(base_dir):
        base_dir="./out"
    else:
        base_dir = f"{base_dir}/out_ML_pytorch"

    if not cfg.output_dir:
        cfg.output_dir = f"{base_dir}/{os.path.basename(cfg_file_name).replace('.yml','')}"
    main_dir = cfg.output_dir

    name = main_dir.strip("/").split("/")[-1]

    best_vloss = 1_000_000.0
    best_vaccuracy = 0.0
    best_epoch = -1
    best_model_name = ""

    loaded_epoch = -1

    n_epochs = cfg.epochs

    assert cfg.learning_rate > 0, "learning_rate must be positive"

    # copy the ML model to the output directory
    saved_ML_model_path = f"{main_dir}/ML_model.py"

    if cfg.load_model or cfg.eval_model:
        # os.system(f"cp {saved_ML_model_path} {file_dir}/../models/ML_model_loaded.py")
        sys.path.append(main_dir)
        print('sys.path',sys.path)
        import ML_model
        
        # ML_model = importlib.import_module(saved_ML_model_path.replace("/", ".").replace(".py", ""))
        # ML_model = importlib.import_module(f"ml_pytorch.models.ML_model_loaded")

        # ML_model = importlib.import_module(f"ml_pytorch.models.{cfg.ML_model}")
    else:
        ML_model = importlib.import_module(f"ml_pytorch.models.{cfg.ML_model}")
        try:
            os.makedirs(main_dir)
        except FileExistsError:
            # ask the user if they want to overwrite the directory
            print(f"Directory {main_dir} already exists")
            if cfg.overwrite:
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

    ML_model_path = f"{file_dir}/../models/{cfg.ML_model}.py"
    # writer = SummaryWriter(f"runs/DNN_trainer_{timestamp}")
    # Create the logger
    logger_file = f"{main_dir}/logger_{name}.log"
    logger = setup_logger(logger_file, cfg.verbosity)

    logger.info(f"Output directory: {main_dir}")

    if not cfg.eval_model and not cfg.load_model:
        logger.info("Copying ML model and config to output directory")
        os.system(f"cp {ML_model_path} {saved_ML_model_path}")
        os.system(f"cp {args.config} {main_dir}")
        # dump cfg in yml file
        cfg_out_file_name = f"{main_dir}/config_parameters.yml"
        OmegaConf.save(cfg, cfg_out_file_name)

    logger.info("=" * 20)
    logger.info("default configs")
    logger.info("cfg:\n - %s", "\n - ".join(str(it) for it in default_cfg.items()))

    logger.info("=" * 20)
    logger.info("args")
    logger.info("args:\n - %s", "\n - ".join(str(it) for it in args.__dict__.items()))

    logger.info("=" * 20)
    logger.info("configs")
    logger.info("cfg:\n - %s", "\n - ".join(str(it) for it in cfg.items()))

    if type(cfg.input_variables) == str:
        logger.info("Get Input variables from dnn_inputs")
        logger.info(cfg.input_variables)
        dnn_input_variables_file=importlib.import_module(f"ml_pytorch.defaults.{cfg.input_variables}")
        
        cfg.input_variables = create_DNN_columns_list(
            cfg.run2, dnn_input_variables_file.dnn_input_variables
        )
        
    input_variables = cfg.input_variables
    logger.info(input_variables)

    early_stopping = cfg.early_stopping
    patience = cfg.patience
    min_delta = cfg.min_delta

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
    ) = load_data(cfg, cfg.seed)
    if cfg.gpus is not None:
        gpus = [int(i) for i in cfg.gpus.split(",")]
        device = torch.device(gpus[0])
    else:
        gpus = None
        device = torch.device("cpu")
    logger.info(f"Using {device} device")

    input_size = X_fts.size(1) - 1

    # Get validation evaluator (if best model by loss or accuracy)
    eval_param = cfg.eval_param
    logger.debug(f"Eval param: {eval_param}")
    assert eval_param in ["loss", "acc"], "eval_param must be loss or acc"

    # Create stopper class
    early_stopper = EarlyStopper(
        logger=logger, patience=patience, min_delta=min_delta, eval_param=eval_param
    )

    # Get model
    model, loss_fn, optimizer, scheduler = ML_model.get_model(
        input_size, device, cfg.learning_rate, cfg.learning_rate_schedule, n_epochs
    )
    num_parameters = get_model_parameters_number(model)

    logger.info(f"Number of parameters: {num_parameters}")

    if gpus is not None and len(gpus) > 1:
        # model becomes `torch.nn.DataParallel` w/ model.module being the original `torch.nn.Module`
        model = torch.nn.DataParallel(model, device_ids=gpus)

    if cfg.load_model or cfg.eval_model:
        checkpoint = torch.load(cfg.load_model if cfg.load_model else cfg.eval_model)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        loaded_epoch = checkpoint["epoch"]
        best_model_name = cfg.load_model if cfg.load_model else cfg.eval_model

        with open(logger_file, "r") as f:
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

    if not cfg.eval_model:
        stop_training = False

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
                cfg,
            )

            logger.info(
                "Time elapsed training: {:.2f}s".format(time.time() - time_epoch)
            )

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
                cfg,
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
            validator = avg_vaccuracy if eval_param == "acc" else avg_vloss
            logger.debug(f"Validator: {validator}")
            if early_stopping and early_stopper.early_stop(validator, epoch):
                logger.info("Stopping early")
                stop_training = True
            if epoch == n_epochs - 1:
                stop_training = True

            logger.info("Total time elapsed: {:.2f}s".format(time.time() - time_epoch))
            if stop_training:
                break

            epoch += 1

        # save the last model
        save_pytorch_model(
            main_dir,
            epoch,
            model,
            optimizer,
        )
        model.to("cpu")

        model_dir = f"{main_dir}/state_dict/"
        export_onnx(
            model,
            model_dir,
            batch_size,
            input_size,
            "cpu",
            f"model_last_epoch_{epoch}",
        )

    if cfg.history:
        # plot the training and validation loss and accuracy
        print("\n\n\n")
        logger.info("Plotting training and validation loss and accuracy")
        train_accuracy, train_loss, val_accuracy, val_loss, lr = read_from_txt(
            logger_file
        )

        plot_history(
            train_accuracy,
            train_loss,
            val_accuracy,
            val_loss,
            main_dir,
            False,
        )
        plot_lr(lr, main_dir, False)

    # load best model
    model.load_state_dict(
        torch.load(best_model_name if not cfg.eval_model else cfg.eval_model)[
            "state_dict"
        ]
    )
    model.eval()

    if cfg.onnx:
        # export the model to ONNX
        print("\n\n\n")
        logger.info("Exporting model to ONNX")
        # move model to cpu
        model.to("cpu")
        export_onnx(
            model,
            best_model_name if not cfg.eval_model else cfg.eval_model,
            batch_size,
            input_size,
            "cpu",
            (
                f"model_best_epoch_{best_epoch}"
                if not cfg.eval_model
                else os.path.basename(cfg.eval_model).replace(".pt", "")
            ),
        )

    model.to(device)
    model.eval()

    if cfg.eval or cfg.eval_model:
        # evaluate model on test_dataset loadining the best model
        logger.info("\n\n\n")
        logger.info("Evaluating best model on test and train dataset")
        logger.info("================================")
        # torch.cuda.empty_cache()

        eval_epoch = loaded_epoch if cfg.eval_model else best_epoch
        logger.info("Training dataset\n")
        score_lbl_array_train, loss_eval_train, accuracy_eval_train = eval_model(
            model,
            training_loader,
            loss_fn,
            "training",
            device,
            eval_epoch,
        )

        logger.info("\n")
        logger.info("Test dataset")
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


        if args.save_numpy:
            # save the score and label arrays
            np.savez(
                f"{main_dir}/score_lbl_array.npz",
                score_lbl_array_train=score_lbl_array_train,
                score_lbl_array_test=score_lbl_array_test,
                train_test_fractions=train_test_fractions,
            )

        # plot the signal and background distributions
        if cfg.histos:
            print("\n\n\n")
            logger.info("Plotting signal and background distributions")
            plot_sig_bkg_distributions(
                score_lbl_array_train,
                score_lbl_array_test,
                main_dir,
                False,
                [],
                # [0.3363, 0.3937],
                train_test_fractions[1],
            )
        if cfg.roc:
            print("\n\n\n")
            logger.info("Plotting ROC curve")
            plot_roc_curve(score_lbl_array_test, main_dir, False)

    # remove ML_model_loaded.py
    if cfg.load_model or cfg.eval_model:
        os.system(f"rm {file_dir}/../models/ML_model_loaded.py")
        logger.info("Removed ML_model_loaded.py")

    logger.info("Saved output in %s" % main_dir)

    # print time in hours and minutes
    logger.info(
        "Total time: %s"
        % str(datetime.timedelta(seconds=int(time.time() - start_time)))
    )


if __name__ == "__main__":
    main()
