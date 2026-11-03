"""
simple_cnn_improved.py -- simple_cnn.py plus the standard tricks that make a
CNN train faster and generalize better on CIFAR-10.

Compared to simple_cnn.py, every change is marked with  # NEW:
    1. Data augmentation (random shifts + horizontal flips) on training images
    2. BatchNorm after every convolution
    3. A deeper network: three stages of two conv layers each
    4. Dropout before the classifier
    5. A higher learning rate with a one-cycle schedule
    6. Bigger batches (128 instead of 32)

It still trains for 10 epochs, the same as simple_cnn.py, so the accuracy
difference comes from these techniques rather than from training longer.
On a GPU, try num_epochs = 30 for a few more points.

As with simple_cnn.py, this is deliberately NOT the project's
CustomCNN/RefCNN design, and the project's Pipeline doesn't allow most of
these changes (its image transforms are fixed). This is for learning.

Run it from the project's src/ folder so it reuses src/data/:

    python simple_cnn_improved.py
"""

import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)


# ---------------------------------------------------------------------------
# 1. Data -- now with augmentation
# ---------------------------------------------------------------------------
# NEW: data augmentation. Every time a training image is loaded, it gets a
# small random change, so across 10 epochs the model sees 10 slightly
# different versions of each image instead of the same pixels 10 times.
# It can't simply memorize the training set anymore; it has to learn what
# actually makes a cat a cat. This is the single biggest fix for
# overfitting on a small dataset like CIFAR-10.
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),

    # NEW: pad the 32x32 image with 4 pixels of zeros on every side
    # (-> 40x40), then cut out a random 32x32 window. Net effect: the image
    # is shifted by up to 4 pixels in a random direction.
    transforms.RandomCrop(32, padding=4),

    # NEW: mirror the image left-to-right half the time. A car facing left
    # is still a car. (No vertical flips: an upside-down horse is not a
    # realistic photo, so that would teach the model the wrong things.)
    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Test images are NOT augmented. We want to measure the model on the real,
# unaltered images, and the same way every time.
test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

train_set = torchvision.datasets.CIFAR10(root="./data", train=True,
                                         download=True, transform=train_transform)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False,
                                        download=True, transform=test_transform)

# NEW: batch_size=128 instead of 32. Larger batches use the hardware more
# efficiently (fewer, bigger matrix multiplications) and give BatchNorm
# better statistics (see below). num_workers=2 loads and augments the next
# batches in background processes while the model trains on this one.
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                           shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                                          shuffle=False, num_workers=2)


# ---------------------------------------------------------------------------
# 2. Model -- deeper, with BatchNorm and Dropout
# ---------------------------------------------------------------------------
def conv_block(in_channels, out_channels):
    """One conv layer + BatchNorm + ReLU. Keeps the height and width."""
    return nn.Sequential(
        # bias=False because the BatchNorm right after it has its own shift
        # term, which makes a conv bias redundant.
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),

        # NEW: BatchNorm. For each channel, it takes the values across the
        # whole batch and rescales them to mean 0 and standard deviation 1,
        # then applies a learned scale and shift (2 learnable numbers per
        # channel). Without it, the typical size of the values flowing
        # through the network drifts as earlier layers' weights change,
        # and every later layer has to keep re-adapting. With it, every
        # layer sees inputs in a stable range, so training is faster, much
        # less sensitive to the learning rate, and deeper networks become
        # practical.
        # BatchNorm behaves differently in model.train() (uses the current
        # batch's statistics) and model.eval() (uses running averages it
        # collected during training) -- this is why calling train()/eval()
        # actually matters in this file.
        nn.BatchNorm2d(out_channels),

        nn.ReLU(),
    )


# NEW: three stages instead of two, with TWO conv layers per stage. Stacking
# two 3x3 convs before pooling lets each stage learn more complex features,
# and the third stage lets the network see larger parts of the image.
model = nn.Sequential(
    #                                              input: (batch, 1, 32, 32)
    conv_block(1, 32),                              # -> (batch, 32, 32, 32)
    conv_block(32, 32),                             # -> (batch, 32, 32, 32)
    nn.MaxPool2d(2),                                # -> (batch, 32, 16, 16)

    conv_block(32, 64),                             # -> (batch, 64, 16, 16)
    conv_block(64, 64),                             # -> (batch, 64, 16, 16)
    nn.MaxPool2d(2),                                # -> (batch, 64, 8, 8)

    conv_block(64, 128),                            # -> (batch, 128, 8, 8)
    conv_block(128, 128),                           # -> (batch, 128, 8, 8)
    nn.MaxPool2d(2),                                # -> (batch, 128, 4, 4)

    nn.Flatten(),                                   # -> (batch, 2048)

    # NEW: Dropout. During training, randomly set 30% of these values to 0
    # (and scale the rest up to compensate), with a different random 30%
    # every batch. The classifier can't rely on any single feature always
    # being there, so it learns to use many features together, which
    # generalizes better. In model.eval() dropout is switched off.
    nn.Dropout(0.3),
    nn.Linear(2048, 256),                           # -> (batch, 256)
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 10),                             # -> (batch, 10) logits
)
model = model.to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model has {num_params:,} learnable parameters")


# ---------------------------------------------------------------------------
# 3. Loss, optimizer, and learning-rate schedule
# ---------------------------------------------------------------------------
loss_fn = nn.CrossEntropyLoss()
num_epochs = 40

# NEW: the learning rate is now set by the scheduler below; the lr given
# here is just a starting value it overrides. weight_decay gently pulls all
# weights toward 0 every step, which discourages extreme weights and
# reduces overfitting (0.01 is AdamW's default; written out here so it's
# visible).
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# NEW: one-cycle learning-rate schedule. Instead of one fixed learning rate:
#   * first ~30% of training: ramp the learning rate UP from small to max_lr
#     (gentle start while the weights are still random),
#   * rest of training: smoothly bring it DOWN to nearly 0
#     (big steps to make fast progress, then small steps to settle into a
#     good solution instead of bouncing around it).
# max_lr = 3e-3 is 30x the 1e-4 used in simple_cnn.py; BatchNorm is what
# makes such a high learning rate stable.
# The schedule needs to know the total number of optimizer steps, so it's
# stepped once per BATCH (see the training loop), not once per epoch.
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-3, epochs=num_epochs, steps_per_epoch=len(train_loader))


# ---------------------------------------------------------------------------
# 4 + 5. Training and evaluation
# ---------------------------------------------------------------------------
for epoch in range(num_epochs):
    start = time.time()

    model.train()       # BatchNorm uses batch statistics, Dropout is on
    total_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = loss_fn(model(images), labels)
        loss.backward()
        optimizer.step()
        scheduler.step()    # NEW: move the learning rate one step along the schedule

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    model.eval()        # BatchNorm uses running averages, Dropout is off
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predicted = model(images).argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = 100 * correct / total

    # Note: with augmentation and dropout on during training, the training
    # loss is measured on HARDER inputs than the test set, so it can look
    # higher than you'd expect compared to the test accuracy.
    print(f"Epoch {epoch + 1:2d}/{num_epochs}   "
          f"train loss {avg_loss:.3f}   test accuracy {accuracy:.1f}%   "
          f"lr {scheduler.get_last_lr()[0]:.2e}   ({time.time() - start:.0f} s)")

torch.save(model.state_dict(), "simple_cnn_improved.pth")
print("Saved weights to simple_cnn_improved.pth")