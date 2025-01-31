import torch
import torch.nn as nn
import torch.nn.functional as F


class AMiL(nn.Module):

    def __init__(self):
        super(AMiL, self).__init__()

        # condense every image into self.L features (further encoding before actual MIL starts)
        self.L = 400
        self.D = 128  # hidden layer size for attention network

        # feature extractor before multiple instance learning starts
        self.ftr_proc = nn.Sequential(
            # nn.Conv2d(2048, 1024, kernel_size=1),
            # nn.ReLU(),
            # nn.Conv2d(1024, 512, kernel_size=1),
            # nn.ReLU(),
            # (500, 512, 300, 2,2)
            nn.Conv2d(512, 300, kernel_size=2),
            # (500, 512, 300, 2,2)
            nn.ReLU(),
            # (500, 300, 200, 2,2)
            nn.Conv2d(300, 200, kernel_size=2),
            # (500, 300, 200, 2,2)
            nn.ReLU(),
            # (500, 200, 100, 2,2)
            nn.Conv2d(200, 100, kernel_size=2),
            # (500, 200, 100, 2,2)
            nn.ReLU(),
            # (500,100,2,2) -> (500,400)
            nn.Flatten()
        )

        # Networks for single attention approach
        # attention network (single attention approach)
        self.attention = nn.Sequential(
            # (128,400)
            nn.Linear(self.L, self.D),
            # (128,400)
            nn.Tanh(),
            # (1,128)
            nn.Linear(self.D, 1)
        )

        # classifier (single attention approach)
        self.classifier = nn.Sequential(
            nn.Linear(self.L, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        # (500,400)
        ft = self.ftr_proc(x)

        # calculate attention
        # (500,1)
        att_raw = self.attention(ft)
        # (1,500)
        att_raw = torch.transpose(att_raw, 1, 0)
        # (1,500)
        att_softmax = F.softmax(att_raw, dim=1)
        # end dim (1,400)
        bag_features = torch.mm(att_softmax, ft)

        prediction = self.classifier(bag_features)

        return prediction

    def get_features(self, x):
        ft = self.ftr_proc(x)

        # calculate attention
        att_raw = self.attention(ft)
        att_raw = torch.transpose(att_raw, 1, 0)

        att_softmax = F.softmax(att_raw, dim=1)
        bag_features = torch.mm(att_softmax, ft)
        return bag_features

    def get_full_prediction(self, x):
        ft = self.ftr_proc(x)

        # calculate attention
        att_raw = self.attention(ft)
        att_raw = torch.transpose(att_raw, 1, 0)

        att_softmax = F.softmax(att_raw, dim=1)
        bag_features = torch.mm(att_softmax, ft)

        prediction = self.classifier(bag_features)

        return prediction, att_softmax, bag_features, ft


# DER MODEL
class AMiLExpandable(AMiL):
    def __init__(self):
        super(AMiLExpandable, self).__init__()

        # Store additional feature extractors in a list
        self.additional_feature_extractors = nn.ModuleList()

        # Auxiliary classifier for DER
        self.aux_classifier = nn.Sequential(
            nn.Linear(self.L, 256),
            nn.ReLU(),
            # Includes one extra class for old concepts
            nn.Linear(256, 8)
        )

    def add_feature_extractor(self, input_channels, output_channels, hidden_layers):
        """
        Add a new feature extractor for the current task with a learnable mask.
        Args:
            input_channels (int): Number of input channels for the feature extractor.
            output_channels (int): Number of output features for the extractor.
            hidden_layers (list): Hidden layer dimensions.
        """
        layers = []
        in_ch = input_channels
        for hidden_ch in hidden_layers:
            layers.append(nn.Conv2d(in_ch, hidden_ch, kernel_size=2))
            layers.append(MaskLayer(hidden_ch))  # Add channel-level mask layer
            layers.append(nn.ReLU())
            in_ch = hidden_ch

        layers.append(nn.Flatten())
        # Adjust based on final spatial dims
        conv_output_size = hidden_layers[-1] * 2 * 2
        layers.append(nn.Linear(conv_output_size, output_channels))

        # Add the feature extractor to the list
        self.additional_feature_extractors.append(nn.Sequential(*layers))

    def update_attention(self):
        """
        Update the attention mechanism dynamically based on the current combined feature size.
        """
        # Calculate the new size of combined features

        combined_feature_size = self.L * \
            (len(self.additional_feature_extractors) + 1)

        # Redefine the attention mechanism
        self.attention = nn.Sequential(
            # Adjust input size dynamically
            nn.Linear(combined_feature_size, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )

    def update_attention_aux(self):
        """
        Update the attention mechanism dynamically based on the current combined feature size.
        """
        # Calculate the new size of combined features

        combined_feature_size = self.L
        # Redefine the attention mechanism
        self.attention = nn.Sequential(
            # Adjust input size dynamically
            nn.Linear(combined_feature_size, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )

    def update_classifier(self):
        """
        Update the classifier dynamically based on the current combined feature size.
        """
        combined_feature_size = self.L * \
            (len(self.additional_feature_extractors) + 1)
        # Keep the number of output classes
        num_classes = self.classifier[-1].out_features

        # Redefine the classifier with the updated input size
        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def update_classifier_aux(self, num_classes):
        """
        Update the auxiliary classifier for the current task.
        Args:
            num_classes (int): Number of classes in the current task + 1 for "other".
        """
        combined_feature_size = self.L

        self.aux_classifier = nn.Sequential(
            nn.Linear(combined_feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)  # Dynamically set the output size
        )

    def forward_main(self, x):
        base_features = self.ftr_proc(x)

        # Collect features from additional extractors with masks
        additional_features = []
        for extractor in self.additional_feature_extractors:
            current_feature_ext = extractor(x)
            additional_features.append(current_feature_ext)

        # Concatenate all features, no chaining !
        combined_features = torch.cat(
            [base_features] + additional_features, dim=1)

        # Apply dynamic attention mechanism
        att_raw = self.attention(combined_features)
        att_raw = torch.transpose(att_raw, 1, 0)
        att_softmax = F.softmax(att_raw, dim=1)
        bag_features = torch.mm(att_softmax, combined_features)

        # Classifier prediction
        prediction = self.classifier(bag_features)
        # Return prediction and the newest feature extractor's output
        newest_features = additional_features[-1] if additional_features else None
        # If newest_features exists, pool it to match bag_features shape
        if newest_features is not None:
            # Pool newest_features using attention
            # Shape: [1, feature_dim]
            newest_features = torch.mm(att_softmax, newest_features)

        return prediction

    def forward_aux(self, x):
        # Collect features from additional extractors with masks
        current_feature_ext = self.additional_feature_extractors[-1]
        current_features = current_feature_ext(x)

        # Apply dynamic attention mechanism
        att_raw = self.attention(current_features)
        att_raw = torch.transpose(att_raw, 1, 0)
        att_softmax = F.softmax(att_raw, dim=1)
        bag_features = torch.mm(att_softmax, current_features)

        # Classifier prediction
        prediction = self.aux_classifier(bag_features)
        return prediction

    def forward(self, x, mode="main"):
        if mode == "main":
            return self.forward_main(x)
        elif mode == "aux":
            return self.forward_aux(x)
        else:
            raise ValueError("Invalid mode. Use 'main' or 'aux'.")

    # can be deleted
    def get_features_main(self, x):
        """Get features from all extractors."""
        base_features = self.ftr_proc(x)

        # Collect features from additional extractors with masks
        additional_features = []
        for extractor in self.additional_feature_extractors:
            current_feature_ext = extractor(x)
            additional_features.append(current_feature_ext)

        # Concatenate all features
        combined_features = torch.cat(
            [base_features] + additional_features, dim=1)

        # Apply dynamic attention mechanism
        att_raw = self.attention(combined_features)
        att_raw = torch.transpose(att_raw, 1, 0)
        att_softmax = F.softmax(att_raw, dim=1)
        bag_features = torch.mm(att_softmax, combined_features)
        return bag_features

    def get_features_aux(self, x):
        self.update_attention_aux()
        # Collect features from additional extractors with masks
        current_feature_ext = self.additional_feature_extractors[-1]
        current_features = current_feature_ext(x)

        # Apply dynamic attention mechanism
        att_raw = self.attention(current_features)
        att_raw = torch.transpose(att_raw, 1, 0)
        att_softmax = F.softmax(att_raw, dim=1)
        bag_features = torch.mm(att_softmax, current_features)
        return bag_features


class MaskLayer(nn.Module):

    def __init__(self, num_channels):
        """
        A custom mask layer for applying a learnable mask to each channel.
        Args:
            num_channels (int): Number of channels to mask.
        """
        super(MaskLayer, self).__init__()
        # Learnable mask parameter for each channel
        self.mask = nn.Parameter(torch.ones(
            num_channels), requires_grad=True)

    def binarize(self):
        """
        Binarize the mask parameters to create a binary mask.
        """
        with torch.no_grad():
            self.mask.data = (
                self.mask.data > 0.5).float()

    def forward(self, x):
        """
        Apply the mask to the input tensor.
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_channels, height, width].
        Returns:
            torch.Tensor: Masked output.
        """
        # Apply the mask channel-wise (reshape mask to [1, num_channels, 1, 1])
        self.binarize()
        return x * self.mask.view(1, -1, 1, 1)
