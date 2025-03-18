import torch
import os
import time
import logging

logger = logging.getLogger(__name__)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def loop_one_batch(
    running_loss,
    tot_loss,
    running_correct,
    tot_correct,
    running_num,
    tot_num,
    count,
    i,
    data,
    model,
    optimizer,
    loss_fn,
    device,
    train,
    time_epoch,
    num_batches,
    epoch_index,
    eval_model,
    all_scores,
    all_labels,
    all_weights,
    type_eval,
):
    inputs, labels, class_weights = data
    inputs = inputs.to(device)
    labels = labels.to(device)
    class_weights = class_weights.to(device)

    event_weights = inputs[:, -1]
    inputs = inputs[:, :-1]
    weights = event_weights * class_weights.flatten()

    # compute the outputs of the model
    outputs = model(inputs)

    if outputs.shape[1] == 1:
        outputs=outputs.flatten()
        y_pred = torch.round(outputs)
        
    else:
        y_pred = outputs.argmax(dim=1)
        labels=labels.type(dtype=torch.long)
        
        
    # Compute the accuracy
    correct = ((y_pred == labels) * weights).sum().item()
    # correct = ((y_pred == labels).view(1, -1).squeeze() * weights).sum().item()

    # Compute the loss and its gradients
    loss = loss_fn(outputs, labels)
    # reshape the loss
    loss = loss.view(1, -1).squeeze()
    # weight the loss
    loss = loss * weights
    # weighted average of the loss
    loss_average = loss.sum() / weights.sum()
    
    if i==50 and epoch_index==0:
        print("outputs", outputs, outputs.shape)
        print("labels", labels, labels.shape)
        print("loss", loss, loss.shape)

    if train:
        # Reset the gradients of all optimized torch.Tensor
        optimizer.zero_grad()
        # Compute the gradients of the loss w.r.t. the model parameters
        loss_average.backward()
        # Adjust learning weights
        optimizer.step()

    # Gather data for reporting
    running_loss += loss_average.item()
    tot_loss += loss_average.item()

    running_correct += correct
    tot_correct += correct

    running_num += weights.sum().item()
    tot_num += weights.sum().item()

    step_prints = max(1, 0.1 * num_batches)

    if i + 1 >= step_prints * count:
        count += 1

        last_loss = running_loss / step_prints  # loss per batch
        last_accuracy = running_correct / running_num  # accuracy per batch
        tb_x = epoch_index * num_batches + i + 1

        logger.info(
            "EPOCH # %d, time %.1f,  %s batch %.1f %% , %s        accuracy: %.4f      //      loss: %.4f"
            % (
                epoch_index,
                time.time() - time_epoch,
                (
                    ("Training" if train else "Validation")
                    if not eval_model
                    else f"Evaluating ({type_eval})"
                ),
                (i + 1) / num_batches * 100,
                f"step {tb_x}" if not eval_model else "",
                last_accuracy,
                last_loss,
            )
        )

        # type = "train" if train else "val"
        # tb_writer.add_scalar(f"Accuracy/{type}", last_accuracy, tb_x)
        # tb_writer.add_scalar(f"Loss/{type}", last_loss, tb_x)

        running_loss = 0.0
        running_correct = 0
        running_num = 0

    if eval_model:
        # Create array of scores and labels
        if i == 0:
            all_scores = outputs
            all_labels = labels
            all_weights = event_weights
        else:
            all_scores = torch.cat((all_scores, outputs))
            all_labels = torch.cat((all_labels, labels))
            all_weights = torch.cat((all_weights, event_weights))

    return (
        running_loss,
        running_correct,
        running_num,
        tot_loss,
        tot_correct,
        tot_num,
        count,
        all_scores,
        all_labels,
        all_weights,
    )


def train_val_one_epoch(
    train,
    epoch_index,
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    time_epoch,
    scheduler,
    main_dir=None,
    best_loss=None,
    best_accuracy=None,
    best_epoch=None,
    eval_param="loss",
    best_model_name=None,
):
    logger.info("Training" if train else "Validation")
    if train: 
        model.train(train)
    else:
        model.eval()

    running_loss = 0.0
    tot_loss = 0.0

    running_correct = 0
    tot_correct = 0

    running_num = 0
    tot_num = 0

    num_batches = len(loader)
    count = 0

    all_scores = None
    all_labels = None
    all_weights = None

    # Loop over the training data
    for i, data in enumerate(loader):
        (
            running_loss,
            running_correct,
            running_num,
            tot_loss,
            tot_correct,
            tot_num,
            count,
            *_,
        ) = loop_one_batch(
            running_loss,
            tot_loss,
            running_correct,
            tot_correct,
            running_num,
            tot_num,
            count,
            i,
            data,
            model,
            optimizer,
            loss_fn,
            device,
            train,
            time_epoch,
            num_batches,
            epoch_index,
            False,
            all_scores,
            all_labels,
            all_weights,
            None,
        )

    if train:
        logger.info(
            "EPOCH # %d, learning rate: %.6f"
            % (epoch_index, optimizer.param_groups[0]["lr"])
        )
        scheduler.step()

    avg_loss = tot_loss / len(loader)
    avg_accuracy = tot_correct / tot_num

    # Track best performance, and save the model state
    if eval_param == "loss":
        evaluator=avg_loss
        best_eval=best_loss
    elif eval_param == "acc":
        evaluator=1-avg_accuracy
        best_eval=1-best_accuracy
    else:
        raise ValueError("Bad evaluator name")
    if not train and evaluator < best_eval:
        best_eval = evaluator
        best_loss = avg_loss
        best_accuracy = avg_accuracy
        best_epoch = epoch_index

        model_dir = f"{main_dir}/models"
        state_dict_dir = f"{main_dir}/state_dict"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(state_dict_dir, exist_ok=True)
        # model_name = f"{model_dir}/model_{epoch_index}.pt"
        state_dict_name = f"{state_dict_dir}/model_{epoch_index}_state_dict.pt"
        checkpoint = {
            "epoch": epoch_index,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, state_dict_name)
        # torch.save(model, model_name)
        best_model_name = state_dict_name

    return (
        avg_loss,
        avg_accuracy,
        best_loss,
        best_accuracy,
        best_epoch,
        best_model_name,
    )


def eval_model(model, loader, loss_fn, type, device, best_epoch):
    # Test the model by running it on the test set
    running_loss = 0.0
    tot_loss = 0.0

    running_correct = 0
    tot_correct = 0

    running_num = 0
    tot_num = 0

    num_batches = len(loader)
    count = 1

    all_scores = None
    all_labels = None
    all_weights = None

    for i, data in enumerate(loader):
        (
            running_loss,
            running_correct,
            running_num,
            tot_loss,
            tot_correct,
            tot_num,
            count,
            all_scores,
            all_labels,
            all_weights,
        ) = loop_one_batch(
            running_loss,
            tot_loss,
            running_correct,
            tot_correct,
            running_num,
            tot_num,
            count,
            i,
            data,
            model,
            None,
            loss_fn,
            device,
            False,
            time.time(),
            num_batches,
            best_epoch,
            True,
            all_scores,
            all_labels,
            all_weights,
            type,
        )

    avg_loss = tot_loss / len(loader)
    avg_accuracy = tot_correct / tot_num

    # concatenate all scores and labels
    all_scores = all_scores.view(-1, 1)
    all_labels = all_labels.view(-1, 1)
    all_weights = all_weights.view(-1, 1)

    score_lbl_tensor = torch.cat((all_scores, all_labels, all_weights), 1)

    # detach the tensor from the graph and convert to numpy array
    score_lbl_array = score_lbl_tensor.cpu().detach().numpy()

    return score_lbl_array, avg_loss, avg_accuracy


def export_onnx(model, model_name, batch_size, input_size, device):

    class ONNXWrappedModel(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model  # Use the trained model

        def forward(self, x):
            logits = self.model(x)  # Get raw logits
            return torch.nn.functional.softmax(logits, dim=1)  # Apply softmax inside ONNX model
    
    if hasattr(model, "export_model"):
        model = model.export_model(model)
    #model = ONNXWrappedModel(model)

    # Export the model to ONNX format
    dummy_input = torch.zeros(batch_size, input_size, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        model_name.replace(".pt", ".onnx"),
        verbose=True,
        export_params=True,
        opset_version=13,
        input_names=["InputVariables"],  # the model's input names
        output_names=["Output"],  # the model's output names
        dynamic_axes={
            "InputVariables": {0: "batch_size"},  # variable length axes
            "Output": {0: "batch_size"},
        },
    )

def create_DNN_columns_list(run2, dnn_input_variables):
    """Create the columns of the DNN input variables
    """
    column_list = []
    for x, y in dnn_input_variables.values():
        # name = x.split(":")[0]
        name_coll=x
        if run2:
            if ":" in name_coll:
                coll, pos = name_coll.split(":")
                column_list.append(f"{coll}Run2_{y}:{pos}")
            elif name_coll != "events":
                column_list.append(f"{name_coll}Run2_{y}")
            elif "sigma" in y: 
                column_list.append(f"{name_coll}_{y}Run2")
            else:
                column_list.append(f"{name_coll}_{y}")
        else:
            column_list.append(f"{name_coll}_{y}")

    return column_list 

if __name__=="__main__":
    from ml_pytorch.defaults.dnn_input_variables import bkg_morphing_dnn_input_variables
    columns = create_DNN_columns_list(True,bkg_morphing_dnn_input_variables)
    for var in columns:
        print(var)
