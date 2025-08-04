import torch
from torch import nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from models import *
import copy
from torch import optim

# from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import *
import matplotlib.pyplot as plt

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


def train(device, model, train_dataloader, criterion, optimizer, scheduler):
    running_loss = 0.0
    running_accuracy = 0.0
    model.train()
    for data, target in tqdm(train_dataloader):
        data = data.to(device)
        target = target.to(device)
        output = model.forward(data)
        # output = output.unsqueeze(0)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        acc = (output.argmax(dim=1) == target).float().mean()
        running_accuracy += acc / len(train_dataloader)
        running_loss += loss.item() / len(train_dataloader)

    return running_loss, running_accuracy


def evaluation(device, model, dataloader, criterion, resnet_features=None):
    model.eval()

    with torch.no_grad():
        test_accuracy = 0.0
        test_loss = 0.0
        for data, target in tqdm(dataloader):
            data = data.to(device)
            target = target.to(device)
            output = model.forward(data)
            loss = criterion(output, target)
            acc = (output.argmax(dim=1) == target).float().mean()
            test_accuracy += acc / len(dataloader)
            test_loss += loss.item() / len(dataloader)

    return test_loss, test_accuracy


def main():
    params = {
        "batch_size": 128,
        "num_workers": 8,
        "dataAug": True,
        "augType": "random",
        "is_distributed": False,
        "normalize": "svhn",
    }  # 128 for 2-layer scaleformer, 64 for 3-layer scaleformer, 16 for 4-layer scaleformer
    lr = 0.00005  # 0.003 for vit, 0.0001 for 2-layer scaleformer, 0.00005 for 4-layer scaleformer
    classes = 10  # 100 for cifar100, 10 for cifar 10,svhn
    scales = 2
    epochs = 50
    depth = 12  # number of blocks
    proj_dim = 768  # proj_dim: 384
    heads = 12
    mlp_ratio = 4.0
    patch_size = 32  # 224 // 7
    num_patches = 49
    attn_drop_out = 0.0
    proj_drop_out = 0.0
    freeze_backbone = False
    backbone = "r50"  # 'r18
    init_values = None  # 1e-5 for layer scale(from CaiT), no layer scale if None
    weight_decay = 1e-4
    model_ver = "scaleformer"  # scaleformer,vit, R50ViTpretrained, R50ViT, originalViT(pretrained)

    model = build_model(
        depth=depth,
        embed_dim=proj_dim,
        num_heads=heads,
        num_classes=classes,
        num_layers=scales,
        num_patches=num_patches,
        proj_dim=proj_dim,
        mlp_ratio=mlp_ratio,
        attn_drop_rate=attn_drop_out,
        proj_drop_rate=proj_drop_out,
        freeze_backbone=freeze_backbone,
        backbone=backbone,
        pretrained=pretrained,
    )
    # model = build_hybrid(num_classes=100,model_ver=model_ver) # R50ViTpretrained

    num_params, total_params = count_parameters(model)
    print(
        "Trainable Parameter: \t%2.1fM" % num_params,
        "Total Parameter: \t%2.1fM" % total_params,
    )

    train_dataloader, val_dataloader, test_dataloader = build_dataset(
        "SVHN", params=params
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=epochs
    )

    model.to(device)
    train_accs = []
    test_accs = []
    best_te_acc = 0.0
    for epoch in range(epochs):

        running_loss, running_accuracy = train(
            device, model, train_dataloader, criterion, optimizer, scheduler
        )
        print(
            f"Epoch : {epoch+1} - acc: {running_accuracy:.4f} - loss : {running_loss:.4f}\n"
        )
        print("lrAfterScheduler:", optimizer.param_groups[0]["lr"])
        train_accs.append(running_accuracy)

        test_loss, test_accuracy = evaluation(device, model, test_dataloader, criterion)
        print(f"test acc: {test_accuracy:.4f} - test loss : {test_loss:.4f}\n")
        test_accs.append(test_accuracy)

        if test_accuracy > best_te_acc:
            best_te_acc = test_accuracy
            # if rank == 0 and best_te_acc >= 0.75:
            if best_te_acc >= 0.75:
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model,
                        "optimizer": optimizer,
                        "scheduler": scheduler,
                        "train_acc": train_accs,
                        "test_acc": test_accs,
                    },
                    f"./save/cifar100{model.name}Scales{model.num_layers}epoch{epoch}acc{best_te_acc}checkpoint.pt",
                )

    train_accs = [tensor.cpu().numpy() for tensor in train_accs]
    test_accs = [tensor.cpu().numpy() for tensor in test_accs]
    epochs = list(range(epochs))
    fig, ax = plt.subplots()
    ax.plot(epochs, train_accs, label="Training Accuracy", marker="o")
    ax.plot(epochs, test_accs, label="Testing Accuracy", marker="x")

    # Annotate the best accuracy on the curve
    best_test_epoch = np.argmax(test_accs)
    best_test_acc = test_accs[best_test_epoch]
    ax.annotate(
        f"Max Test Acc: {best_test_acc:.2%}",
        xy=(best_test_epoch, best_test_acc),
        xytext=(best_test_epoch, best_test_acc + 5),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="orange"),
    )
    ax.annotate(
        f"Max Train Acc: {train_accs[best_test_epoch]:.2%}",
        xy=(best_test_epoch + 1, train_accs[best_test_epoch]),
        xytext=(0, -15),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color="blue",
    )

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and Testing Accuracy")
    ax.legend()
    plt.savefig(
        "./save/cifar100"
        + model.name
        + "Scales"
        + str(model.num_layers)
        + "lr"
        + str(lr)
        + "accuracy_plot.png"
    )
    plt.show()


if __name__ == "__main__":
    main()
