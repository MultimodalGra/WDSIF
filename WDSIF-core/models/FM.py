import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightLearningModule(nn.Module):

    def __init__(self, input_dim):
        super(WeightLearningModule, self).__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, img_feats, img_feats_n_top, img_feats_n_top_sm):

        combined_feats = torch.cat(
            [img_feats, img_feats_n_top, img_feats_n_top_sm], dim=1
        )
        weights = self.fc(combined_feats)
        alpha = weights[:, 0].unsqueeze(1)
        beta = weights[:, 1].unsqueeze(1)
        return alpha, beta


class FM(nn.Module):

    def __init__(self, feature_dim=3):
        super().__init__()

        self.init_layers()

        self.weight_learner = WeightLearningModule(input_dim=feature_dim * 3)

        self.get_parameters()

    def init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.con1 = nn.Conv2d(9, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in1 = nn.InstanceNorm2d(
            64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False
        )
        self.con2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in2 = nn.InstanceNorm2d(
            64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False
        )
        self.con3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in3 = nn.InstanceNorm2d(
            64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False
        )
        self.con4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in4 = nn.InstanceNorm2d(
            64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False
        )
        self.con5 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in5 = nn.InstanceNorm2d(
            64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False
        )
        self.con6 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.in6 = nn.InstanceNorm2d(
            64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False
        )
        self.con7 = nn.Conv2d(64, 6, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def get_parameters(self):
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.net_parameters = {"Total": total_num, "Trainable": trainable_num}
        print(
            f"FM (Learnable Weights) - Total: {total_num:,}, Trainable: {trainable_num:,}"
        )

    def forward(self, input_image, model1_output, model2_output):

        B, C, H, W = input_image.shape

        fusion_input = torch.cat([input_image, model1_output, model2_output], dim=1)

        h = self.relu(self.in1(self.con1(fusion_input)))
        h = self.relu(self.in2(self.con2(h)))
        h = self.relu(self.in3(self.con3(h)))
        h = self.relu(self.in4(self.con4(h)))
        h = self.relu(self.in5(self.con5(h)))
        h = self.relu(self.in6(self.con6(h)))
        h = self.con7(h)
        attention_weights = self.sigmoid(h)

        weight1, weight2 = torch.split(attention_weights, 3, dim=1)

        img_feats = self.feature_extractor(input_image)
        m1_feats = self.feature_extractor(model1_output)
        m2_feats = self.feature_extractor(model2_output)

        alpha, beta = self.weight_learner(img_feats, m1_feats, m2_feats)

        alpha = alpha.view(B, 1, 1, 1)
        beta = beta.view(B, 1, 1, 1)

        output = alpha * torch.mul(weight1, model1_output) + beta * torch.mul(
            weight2, model2_output
        )

        return output
