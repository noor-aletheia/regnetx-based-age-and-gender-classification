import torch
import torch.nn as nn
import torchvision.models as models
import logging
from config import Config
from focal_loss import FocalLoss

logger = logging.getLogger(__name__)

class DualHeadResNet(nn.Module):
    """ResNet with dual classification heads for age and gender prediction"""
    def __init__(self, model_name: str, num_age_classes: int, num_gender_classes: int, pretrained: bool = True):
        super(DualHeadResNet, self).__init__()
        self.model_name = model_name
        self.num_age_classes = num_age_classes
        self.num_gender_classes = num_gender_classes
        self.backbone, self.feature_dim = self._create_backbone(model_name, pretrained)
        self.age_head = self._create_classification_head(self.feature_dim, num_age_classes, 'age')
        self.gender_head = self._create_classification_head(self.feature_dim, num_gender_classes, 'gender')
        self._initialize_heads()
        logger.info(f"Created {model_name} with {self.feature_dim} features")
        logger.info(f"Age classes: {num_age_classes}, Gender classes: {num_gender_classes}")

    def _create_backbone(self, model_name: str, pretrained: bool):
        if model_name == 'resnet34':
            backbone = models.resnet34(pretrained=pretrained)
        elif model_name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        # Remove the final fully connected layer
        modules = list(backbone.children())[:-2]  # Remove avgpool and fc
        backbone = nn.Sequential(*modules)
        feature_dim = backbone[-1].out_channels if hasattr(backbone[-1], 'out_channels') else 512
        return backbone, feature_dim

    def _create_classification_head(self, input_dim: int, num_classes: int, head_name: str):
        if head_name == 'age':
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        else:
            return nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(0.4),
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )

    def _initialize_heads(self):
        for head in [self.age_head, self.gender_head]:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen for Phase 1 training")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen for Phase 2 training")

    def get_trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        age_logits = self.age_head(features)
        gender_logits = self.gender_head(features)
        return {'age': age_logits, 'gender': gender_logits}

class ResNetModelFactory:
    SUPPORTED_MODELS = ['resnet34', 'resnet50']

    @staticmethod
    def create_model(config: Config, model_name: str) -> DualHeadResNet:
        if model_name not in ResNetModelFactory.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {ResNetModelFactory.SUPPORTED_MODELS}")
        num_age_classes = int(config.get('dataset.age_classes'))
        num_gender_classes = int(config.get('dataset.gender_classes', 2))
        pretrained = config.get('models.pretrained', True)
        model = DualHeadResNet(
            model_name=model_name,
            num_age_classes=num_age_classes,
            num_gender_classes=num_gender_classes,
            pretrained=pretrained
        )
        return model

    @staticmethod
    def get_model_info(model: DualHeadResNet):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            'model_name': model.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_dim': model.feature_dim,
            'age_classes': model.num_age_classes,
            'gender_classes': model.num_gender_classes
        }

    @staticmethod
    def print_model_summary(model: DualHeadResNet):
        info = ResNetModelFactory.get_model_info(model)
        logger.info(f"\n=== Model Summary: {info['model_name']} ===")
        logger.info(f"Total parameters: {info['total_parameters']:,}")
        logger.info(f"Trainable parameters: {info['trainable_parameters']:,}")
        logger.info(f"Feature dimension: {info['feature_dim']}")
        logger.info(f"Age classes: {info['age_classes']}")
        logger.info(f"Gender classes: {info['gender_classes']}")

    @staticmethod
    def list_available_models():
        logger.info("Available models:")
        for i, model in enumerate(ResNetModelFactory.SUPPORTED_MODELS, 1):
            logger.info(f"  {i:2d}. {model}")
        return ResNetModelFactory.SUPPORTED_MODELS
