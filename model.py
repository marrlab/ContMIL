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
        # Create new feature extractor
        new_extractor = nn.Sequential(*layers)

        # **Initialize with previous extractor's weights (if available)**
        if self.additional_feature_extractors:
            # Get last extractor
            prev_extractor = self.additional_feature_extractors[-1]
            with torch.no_grad():
                for new_layer, prev_layer in zip(new_extractor, prev_extractor):
                    # Copy weights if the layer types match
                    if isinstance(new_layer, nn.Conv2d) and isinstance(prev_layer, nn.Conv2d):
                        new_layer.weight.data.copy_(prev_layer.weight.data)
                        new_layer.bias.data.copy_(prev_layer.bias.data)
                    elif isinstance(new_layer, nn.Linear) and isinstance(prev_layer, nn.Linear):
                        new_layer.weight.data.copy_(prev_layer.weight.data)
                        new_layer.bias.data.copy_(prev_layer.bias.data)

        # Add the feature extractor to the list
        self.additional_feature_extractors.append(new_extractor)

    def update_attention(self, mode):
        """
        Update the attention mechanism dynamically based on the current combined feature size.
        """
        # Calculate the new size of combined features

        combined_feature_size = self.L * \
            (len(self.additional_feature_extractors) +
             1) if mode == 'main' else self.L

        # Redefine the attention mechanism
        self.attention = nn.Sequential(
            # Adjust input size dynamically
            nn.Linear(combined_feature_size, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )

    def update_classifier(self, mode, num_classes=None, reset=False):
        """
        Update the classifier dynamically based on the current combined feature size.
        - If `reset=True`, the classifier is fully reinitialized (used for retraining).
        - Otherwise, it inherits old feature weights for continual learning.

        Args:
            mode (str): Determines feature size ('main' or 'aux').
            num_classes (int, optional): Number of output classes.
            reset (bool): If True, fully reinitializes the classifier (for retraining).
        """
        combined_feature_size = self.L * \
            (len(self.additional_feature_extractors) +
             1) if mode == 'main' else self.L

        if not num_classes:
            # Keep number of output classes
            num_classes = self.classifier[-1].out_features

        # Create a new classifier
        new_classifier = nn.Sequential(
            nn.Linear(combined_feature_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        # Only inherit weights if reset=False (continual learning)
        if not reset:
            with torch.no_grad():
                old_classifier = self.classifier
                # Previous feature size
                old_input_size = old_classifier[0].in_features

                # Transfer weights for overlapping dimensions
                new_classifier[0].weight[:,
                                         :old_input_size] = old_classifier[0].weight
                new_classifier[0].bias = old_classifier[0].bias

                new_classifier[2].weight = old_classifier[2].weight
                new_classifier[2].bias = old_classifier[2].bias

        # Assign the new classifier
        self.classifier = new_classifier

    def forward_main(self, x):
        base_features = self.ftr_proc(x)

        # Collect features from additional extractors with masks
        additional_features = []
        for i, extractor in enumerate(self.additional_feature_extractors):
            current_feature_ext = extractor(x)

            # Ensure previous extractors do not update their weights
            if i < len(self.additional_feature_extractors) - 1:
                current_feature_ext = current_feature_ext.detach()
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
    def get_features(self, x, mode):
        """Get features from all extractors."""

        if mode == 'aux':
            self.update_attention(mode='aux')
            # Collect features from additional extractors with masks
            current_feature_ext = self.additional_feature_extractors[-1]
            combined_features = current_feature_ext(x)

        elif mode == 'main':
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
