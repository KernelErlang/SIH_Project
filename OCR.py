import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CHARACTERS = 36
BLANK_ID = 0
NUM_CLASSES = NUM_CHARACTERS + 1


class PlateCRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        self.sequence_pool = nn.AdaptiveAvgPool2d((1, None))

        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.classifier = nn.Linear(512, NUM_CLASSES)

    def forward(self, images):
        features = self.feature_extractor(images)

        features = self.sequence_pool(features)

        features = features.squeeze(2)

        features = features.permute(0, 2, 1)

        sequence_output, _ = self.lstm(features)

        logits = self.classifier(sequence_output)

        log_probabilities = F.log_softmax(logits, dim=2)

        return log_probabilities.permute(1, 0, 2)


if __name__ == "__main__":
    model = PlateCRNN()

    test_images = torch.randn(16, 1, 80, 320)
    output = model(test_images)

    print(f"Input image batch shape: {test_images.shape}")
    print(f"Model output shape: {output.shape}")
    print("Expected format: [time_steps, batch_size, character_classes]")