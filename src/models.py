import torch
import torch.nn as nn
import torchvision.models as models
import os
from typing import Dict, Any, Tuple
import logging
from config import Config
from focal_loss import FocalLoss

# Import custom RegNet implementation
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from regnet import get_regnet, REGNET_MODEL_ZOO
from reglayers import WrappedModel

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
        
        self.backbone = self._create_backbone(model_name, pretrained)
        
        self.feature_dim = self._get_feature_dim()
        
        self.age_head = self._create_classification_head(self.feature_dim, num_age_classes, 'age')
        self.gender_head = self._create_classification_head(self.feature_dim, num_gender_classes, 'gender')
        
        self._initialize_heads()
        
        logger.info(f"Created {model_name} with {self.feature_dim} features")
        logger.info(f"Age classes: {num_age_classes}, Gender classes: {num_gender_classes}")
    
    def _create_backbone(self, model_name: str, pretrained: bool):
        """Create the backbone network using unified RegNet interface"""
        try:
            model_mapping = {
                'regnet_1600m': 'regnet_1600m',
                'regnet_200m': 'regnet_200m',
                'regnet_400m': 'regnet_400m',
                'regnet_600m': 'regnet_600m',
                'regnet_800m': 'regnet_800m',
                'regnet_3200m': 'regnet_3200m',
                'regnet_6400m': 'regnet_6400m',
            }
            if model_name not in model_mapping:
                raise ValueError(f"Unsupported model: {model_name}")
            regnet_key = model_mapping[model_name]
            full_model = get_regnet(regnet_key, pretrained=pretrained)
            backbone_modules = []
            for name, module in full_model.named_children():
                if name != 'head':
                    backbone_modules.append(module)
            backbone = nn.Sequential(*backbone_modules)
            logger.info(f"Created RegNet backbone for {model_name} using get_regnet")
            return backbone
        except Exception as e:
            logger.error(f"Failed to create RegNet backbone {model_name}: {e}")
            raise ValueError(f"Failed to create RegNet backbone: {model_name}")
    

    
    def _get_feature_dim(self) -> int:
        """Get the feature dimension of the backbone from REGNET_MODEL_ZOO or auto-detect."""
        regnet_key = self.model_name
        if regnet_key in REGNET_MODEL_ZOO:
            feature_dim = REGNET_MODEL_ZOO[regnet_key]['feature_dim']
            logger.info(f"Using feature dimension from REGNET_MODEL_ZOO for {self.model_name}: {feature_dim}")
            return feature_dim
        test_input = torch.randn(1, 3, 224, 224)
        try:
            with torch.no_grad():
                features = self.backbone(test_input)
                if len(features.shape) == 4:
                    features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                feature_dim = features.view(features.size(0), -1).shape[1]
            logger.info(f"Auto-detected feature dimension for {self.model_name}: {feature_dim}")
            return feature_dim
        except Exception as e:
            logger.warning(f"Could not detect feature dimension for {self.model_name}, using default 912")
            return 912
    
    def _create_classification_head(self, input_dim: int, num_classes: int, head_name: str):
        """Create a classification head"""
        if head_name == 'age':
            return nn.Sequential(
                nn.AvgPool2d(kernel_size=7, stride=1),
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
                nn.AvgPool2d(kernel_size=7, stride=1),
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
        features = self.backbone(x)
        
        if len(features.shape) == 2:  # Already flattened [B, Features]
            age_logits = self.age_head[2:](features)  # Skip AdaptiveAvgPool2d and Flatten
            gender_logits = self.gender_head[2:](features)
        else:  # [B, C, H, W] format
            age_logits = self.age_head(features)
            gender_logits = self.gender_head(features)
        
        return {
            'age': age_logits,
            'gender': gender_logits
        }


class ModelFactory:
    """Factory class for creating RegNetX models"""
    
    SUPPORTED_MODELS = [
        'regnet_1600m', 'regnet_200m', 'regnet_400m', 'regnet_600m', 'regnet_800m', 'regnet_3200m', 'regnet_6400m'
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
        
        num_age_classes = int(config.get('dataset.age_classes'))
        num_gender_classes = int(config.get('dataset.gender_classes', 2))
        pretrained = config.get('models.pretrained', True)
        pretrained_model_path = config.get('models.pretrained_model_path', None)
        
        model = DualHeadRegNetX(
            model_name=model_name,
            num_age_classes=num_age_classes,
            num_gender_classes=num_gender_classes,
            pretrained=pretrained
        )
        
        # Load custom pretrained weights if provided
        if pretrained_model_path:
            ModelFactory.load_pretrained_weights(model, pretrained_model_path, num_age_classes)
        
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
    def load_pretrained_weights(model: DualHeadRegNetX, pretrained_path: str, target_age_classes: int):
        """
        Load pretrained weights from another model with potentially different number of classes
        
        Args:
            model: Target model to load weights into
            pretrained_path: Path to pretrained model file
            target_age_classes: Number of age classes in target model
        """
        logger.info(f"Loading pretrained weights from: {pretrained_path}")
        
        # Load pretrained state dict
        pretrained_state = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different state dict formats (raw model vs saved with metadata)
        if 'model_state_dict' in pretrained_state:
            pretrained_weights = pretrained_state['model_state_dict']
        elif 'state_dict' in pretrained_state:
            pretrained_weights = pretrained_state['state_dict'] 
        else:
            pretrained_weights = pretrained_state
        
        # Get current model state
        current_state = model.state_dict()
        
        # Copy compatible weights
        loaded_keys = []
        skipped_keys = []
        
        for key, pretrained_weight in pretrained_weights.items():
            if key in current_state:
                current_weight = current_state[key]
                
                # Check if shapes match
                if pretrained_weight.shape == current_weight.shape:
                    current_state[key] = pretrained_weight
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch: {pretrained_weight.shape} vs {current_weight.shape})")
            else:
                skipped_keys.append(f"{key} (not found in target model)")
        
        # Load the updated state dict
        model.load_state_dict(current_state)
        
        logger.info(f"Successfully loaded {len(loaded_keys)} parameter groups")
        logger.info(f"Skipped {len(skipped_keys)} parameter groups due to incompatibility:")
        for key in skipped_keys[:5]:  # Show first 5 skipped keys
            logger.info(f"  - {key}")
        if len(skipped_keys) > 5:
            logger.info(f"  ... and {len(skipped_keys) - 5} more")
        
        # Special handling for classification heads
        age_head_loaded = any('age_classifier' in key for key in loaded_keys)
        gender_head_loaded = any('gender_classifier' in key for key in loaded_keys)
        
        if not age_head_loaded:
            logger.info("Age classifier head will be randomly initialized (different number of classes)")
        if not gender_head_loaded:
            logger.info("Gender classifier head will be randomly initialized")
    
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

    loss_type = config.get('training.loss_type', 'cross_entropy')
    if loss_type == 'focal':
        age_loss_fn = FocalLoss(alpha=age_class_weights, gamma=2.0)
    else:
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