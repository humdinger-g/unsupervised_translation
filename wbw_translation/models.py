from typing import Union, Tuple
import torch
from torch import nn

class ProcrusterMapping:
    def __init__(self,
                 x_embeds: torch.Tensor, 
                 y_embeds: torch.Tensor):
        
        shared = y_embeds.T @ x_embeds
        U, S, V = torch.svd(shared)
        self.W = V @ U.T


class LinearModel(nn.Module):

    def __init__(self,
                 shape: int = 300, 
                 init: Union[str, Tuple[torch.Tensor, torch.Tensor]] = 'eye'):
        
        super().__init__()
        self.shape = shape
        self.Wxy = nn.Linear(shape, shape, bias=False)
        self.Wyx = nn.Linear(shape, shape, bias=False)

        if init == 'eye':
            nn.init.eye_(self.Wxy.weight)
            nn.init.eye_(self.Wyx.weight)
        else:
            self.Wxy.weight = nn.Parameter(init[0])
            self.Wyx.weight = nn.Parameter(init[1])

    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        return self.Wxy(x), self.Wyx(y)
    
    def cycle(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs cycle translation for both souce and target embeddings: 
            X -> Y -> X and Y -> X -> Y

        Args:
            x (torch.Tensor): source embeddings
            y (torch.Tensor): target embeddings

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (cycled source embeddings, cycled target embeddings)
        """
        return self.Wyx(self.Wxy(x)), self.Wxy(self.Wyx(y))
    

class NonLinearModel(nn.Module):

    def __init__(self, 
                 shape: int, 
                 hidden: int = 1024, 
                 activation: nn.Module = nn.ReLU):
        
        super().__init__()
        self.Wxy = nn.Sequential(
            nn.Linear(shape, hidden, bias=False),
            activation(),
            nn.Linear(hidden, shape, bias=False)
        )
        self.Wyx = nn.Sequential(
            nn.Linear(shape, hidden, bias=False),
            activation(),
            nn.Linear(hidden, shape, bias=False)
        )

        nn.init.eye_(self.Wxy[0].weight)
        nn.init.eye_(self.Wxy[2].weight)
        nn.init.eye_(self.Wyx[0].weight)
        nn.init.eye_(self.Wyx[2].weight)
            
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.Wxy(x), self.Wyx(y)
    
    def cycle(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs cycle translation for both souce and target embeddings: 
            X -> Y -> X and Y -> X -> Y

        Args:
            x (torch.Tensor): source embeddings
            y (torch.Tensor): target embeddings

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (cycled source embeddings, cycled target embeddings)
        """
        return self.Wyx(self.Wxy(x)), self.Wxy(self.Wyx(y))


class InferenceModel(nn.Module):
    def __init__(self, W_init):
        super().__init__()
        self.W_init = W_init
        self.W = nn.Linear(W_init.shape[0], W_init.shape[0], bias=False)
        self.W.weight = nn.Parameter(self.W_init)

    def forward(self, x):
        return self.W(x)



def SplitModels(model: LinearModel
            ) -> Tuple[InferenceModel, InferenceModel]:
    """
    model translates from X to Y and from Y to X
    using model.Wxy and model.Wyx respectively
    This function splits this model into two separate with forward method
    that completes translation in a single direction


    Args:
        model (LinearModel): trained LinearModel

    Returns:
        Tuple[InferenceModel, InferenceModel]: model_xy and model_yx that
            perform translations from X to Y and from Y to X respectively
    """
    
    model_xy = InferenceModel(model.Wxy.weight)
    model_yx = InferenceModel(model.Wyx.weight)

    return model_xy, model_yx


def CombineModels(model1: InferenceModel,
                  model2: InferenceModel
            ) -> InferenceModel:
    """
    Combines two models to transitively translate from model1's source language 
    to model2's target language

    Args:
        model1 (InferenceModel): model translating from X to Y
        model2 (InferenceModel): model translating from Y to Z

    Returns:
        InferenceModel: model translating from X to Z
    """

    Wxy = model1.W.weight
    Wyz = model2.W.weight
    W = Wxy.T @ Wyz.T
    model = InferenceModel(W.T)
    
    return model
