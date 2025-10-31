"""
Model architectures for age and gender prediction using RegNetX variants
"""
import torch
import torch.nn as nn
import torchvision.models as models
import timm
from typing import Dict, Any, Tuple
import logging
from config import Config

logger = logging.getLogger(__name__)

class DualHeadRegNetX(nn.Module):
    """RegNetX with dual classification heads for age and gender prediction"""
    
    def __init__(self, model_name: str, num_age_classes: int, num_gender_classes: int, pretrained: bool = True):
        """
        Initialize dual-head MobileNet model
        
        Args:
            model_name: Name of the MobileNet variant 
            num_age_classes: Number of age classes
            num_gender_classes: Number of gender classes
            pretrained: Whether to use pretrained weights
        """
        super(DualHeadRegNetX, self).__init__()
        
        self.model_name = model_name
        self.num_age_classes = num_age_classes
        self.num_gender_classes = num_gender_classes
        
        # Load base model
        self.backbone = self._create_backbone(model_name, pretrained)
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Create classification heads
        self.age_head = self._create_classification_head(self.feature_dim, num_age_classes, 'age')
        self.gender_head = self._create_classification_head(self.feature_dim, num_gender_classes, 'gender')
        
        # Initialize custom layers
        self._initialize_heads()
        
        logger.info(f"Created {model_name} with {self.feature_dim} features")
        logger.info(f"Age classes: {num_age_classes}, Gender classes: {num_gender_classes}")
    
    def _create_backbone(self, model_name: str, pretrained: bool):
        """Create the backbone network"""
        # Use timm models for RegNetX variants
        try:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)  # num_classes=0 removes classifier
            backbone = model
            logger.info(f"Using timm RegNetX model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load timm RegNetX model {model_name}: {e}")
            raise ValueError(f"Unsupported RegNetX model: {model_name}")
        
        return backbone
    
    def _get_feature_dim(self) -> int:
        """Get the feature dimension of the backbone"""
        # Known feature dimensions for RegNetX models
        regnetx_feature_dims = {
            'regnetx_006': 528,
            'regnetx_008': 672,
            'regnetx_016': 912
        }
        
        if self.model_name in regnetx_feature_dims:
            feature_dim = regnetx_feature_dims[self.model_name]
            logger.info(f"Using known feature dimension for {self.model_name}: {feature_dim}")
            return feature_dim
        
        # For unknown models, detect feature dimension dynamically
        test_input = torch.randn(1, 3, 224, 224)
        try:
            with torch.no_grad():
                features = self.backbone(test_input)
                if len(features.shape) == 4:  # [B, C, H, W]
                    # Apply global average pooling
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                feature_dim = features.view(features.size(0), -1).shape[1]
            logger.info(f"Detected feature dimension for {self.model_name}: {feature_dim}")
            return feature_dim
        except Exception as e:
            logger.warning(f"Could not detect feature dimension for {self.model_name}, using default 528")
            return 528
    
    def _create_classification_head(self, input_dim: int, num_classes: int, head_name: str):
        """Create a classification head"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def _initialize_heads(self):
        """Initialize the classification heads with Xavier initialization"""
        for head in [self.age_head, self.gender_head]:
            for module in head.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.constant_(module.bias, 0)
    
    def freeze_backbone(self):
        """Freeze all backbone parameters for Phase 1 training"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen for Phase 1 training")
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters for Phase 2 training"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen for Phase 2 training")
    
    def get_trainable_parameters(self):
        """Get only trainable parameters"""
        return [param for param in self.parameters() if param.requires_grad]
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Dictionary with 'age' and 'gender' predictions
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Handle different output formats from timm vs torchvision
        if len(features.shape) == 2:  # Already flattened [B, Features]
            # For timm models with num_classes=0, features are already pooled and flattened
            age_logits = self.age_head[2:](features)  # Skip AdaptiveAvgPool2d and Flatten
            gender_logits = self.gender_head[2:](features)
        else:  # [B, C, H, W] format
            # For torchvision models, need to apply pooling
            age_logits = self.age_head(features)
            gender_logits = self.gender_head(features)
        
        return {
            'age': age_logits,
            'gender': gender_logits
        }


class ModelFactory:
    """Factory class for creating RegNetX models"""
    
    # RegNetX model variants from timm
    SUPPORTED_MODELS = [
        # RegNetX models
        'regnetx_006', 'regnetx_008', 'regnetx_016'
    ]
    
    @staticmethod
    def create_model(config: Config, model_name: str) -> DualHeadRegNetX:
        """
        Create a model based on configuration
        
        Args:
            config: Configuration object
            model_name: Name of the model to create
            
        Returns:
            Initialized model
        """
        if model_name not in ModelFactory.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {ModelFactory.SUPPORTED_MODELS}")
        
        num_age_classes = int(config.get('dataset.age_classes', 8))
        num_gender_classes = int(config.get('dataset.gender_classes', 2))
        pretrained = config.get('models.pretrained', True)
        
        model = DualHeadRegNetX(
            model_name=model_name,
            num_age_classes=num_age_classes,
            num_gender_classes=num_gender_classes,
            pretrained=pretrained
        )
        
        return model
    
    @staticmethod
    def get_model_info(model: DualHeadRegNetX) -> Dict[str, Any]:
        """Get model information"""
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
    def print_model_summary(model: DualHeadRegNetX):
        """Print model summary"""
        info = ModelFactory.get_model_info(model)
        
        logger.info(f"\n=== Model Summary: {info['model_name']} ===")
        logger.info(f"Total parameters: {info['total_parameters']:,}")
        logger.info(f"Trainable parameters: {info['trainable_parameters']:,}")
        logger.info(f"Feature dimension: {info['feature_dim']}")
        logger.info(f"Age classes: {info['age_classes']}")
        logger.info(f"Gender classes: {info['gender_classes']}")
    
    @staticmethod
    def list_available_models():
        """List all available models"""
        logger.info("Available models:")
        for i, model in enumerate(ModelFactory.SUPPORTED_MODELS, 1):
            logger.info(f"  {i:2d}. {model}")
        return ModelFactory.SUPPORTED_MODELS


def create_loss_functions(config: Config, device: torch.device, 
                         age_class_weights: torch.Tensor = None, 
                         gender_class_weights: torch.Tensor = None) -> Tuple[nn.Module, nn.Module]:
    """
    Create loss functions for age and gender prediction
    
    Args:
        config: Configuration object
        device: Device to put tensors on
        age_class_weights: Optional class weights for age classification
        gender_class_weights: Optional class weights for gender classification
        
    Returns:
        Tuple of (age_loss_fn, gender_loss_fn)
    """
    # Check for alternative manual loss weights in config
    manual_age_weights = config.get('loss_weights.age', None)
    manual_gender_weights = config.get('loss_weights.gender', None)

    if manual_age_weights is not None:
        age_class_weights = torch.tensor(manual_age_weights, dtype=torch.float32, device=device)
    elif age_class_weights is not None:
        age_class_weights = age_class_weights.to(device)
    else:
        age_class_weights = None

    if manual_gender_weights is not None:
        gender_class_weights = torch.tensor(manual_gender_weights, dtype=torch.float32, device=device)
    elif gender_class_weights is not None:
        gender_class_weights = gender_class_weights.to(device)
    else:
        gender_class_weights = None

    # Only use standard CrossEntropyLoss
    age_loss_fn = nn.CrossEntropyLoss(weight=age_class_weights)
    gender_loss_fn = nn.CrossEntropyLoss(weight=gender_class_weights)
    return age_loss_fn, gender_loss_fn


def create_optimizer(model: DualHeadRegNetX, config: Config) -> torch.optim.Optimizer:
    """
    Create optimizer for the model
    
    Args:
        model: The model to optimize
        config: Configuration object
        
    Returns:
        Configured optimizer
    """
    optimizer_name = config.get('training.optimizer', 'AdamW')
    learning_rate = float(config.get('training.phase1.learning_rate', 1e-3))
    weight_decay = float(config.get('training.phase1.weight_decay', 1e-4))
    
    # Get only trainable parameters
    trainable_params = model.get_trainable_parameters()
    
    if optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Config):
    """
    Create learning rate scheduler
    
    Args:
        optimizer: The optimizer to schedule
        config: Configuration object
        
    Returns:
        Configured scheduler
    """
    scheduler_name = config.get('training.scheduler', 'ReduceLROnPlateau')
    
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler_params = config.get('training.scheduler_params', {})
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(scheduler_params.get('factor', 0.5)),
            patience=int(scheduler_params.get('patience', 3)),
            min_lr=float(scheduler_params.get('min_lr', 1e-6))
        )
    elif scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.1
        )
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
            eta_min=1e-6
        )
    else:
        scheduler = None
        logger.warning(f"Unknown scheduler: {scheduler_name}, no scheduler will be used")
    
    return scheduler