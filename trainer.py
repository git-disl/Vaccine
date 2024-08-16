from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import time
import torch
import collections
from packaging import version
from torch.distributions import Categorical
import torch.nn as nn

from transformers import Trainer
from transformers import logging
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import (
    get_parameter_names,
)
from transformers.utils import (
    is_sagemaker_mp_enabled
)

from transformers.models.llama.modeling_llama import LlamaAttention,LlamaMLP
from transformers.models.opt.modeling_opt import OPTAttention

import copy

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)


def get_leaf_modules_with_grad(module):
    # # print([name for name,param  in module.named_parameters()])
    # if len(list(module.children())) == 0 and any(p.requires_grad for p in module.parameters()) and "lora_B" in module._get_name():
    #     return [module]
    # else:
    #     return [submodule for child in module.children() for submodule in get_leaf_modules_with_grad(child)]
    module_list= []
    for name, module in module.named_modules():
    #     if "lora_B" in name and "v_proj" in name and len(list(module.children())) == 0:
    #         module_list+= [module]
    # or isinstance(module, LlamaMLP)
        if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention):
            module_list+= [module]
    # # print(module_list)
    return module_list
            
            
class VaccineTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss 
        # print("calling sam")
        self.vaccine_state = {}
        self.vaccine_state ["hooks"] = []
        self.vaccine_state ["gradient"] = {}
        self.pre_first_step(model)
        step()
        self.after_first_step(model)
        model.zero_grad()
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)
        return loss.detach() / self.args.gradient_accumulation_steps

    @torch.no_grad()
    def pre_first_step(self, model ):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.vaccine_state["gradient"][module] = grad_output[0].detach().clone()/self.args.gradient_accumulation_steps
            # print(grad_output[0])
            
        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list
            
        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.vaccine_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.vaccine_state["hooks"])
            
    
    
    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.vaccine_state["gradient"][module]
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            output[0].data =output[0] + perturbation
            # print(perturbation.shape)
            # print(output.shape)
            return output
           
        
        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
        
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.vaccine_state["hooks"])
        
    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []
        
        # print(self.vaccine_state["gradient"].items())
        grad_norm = self._grad_norm(self.vaccine_state["gradient"])
        # logging.info(grad_norm)
        # logging.info("norm{}".format(grad_norm))
        for module in self.vaccine_state["gradient"]:
            # grad_norm = self._grad_norm(self.vaccine_state["gradient"][module])
            grad = self.vaccine_state["gradient"][module]
            scale = self. args. rho  / (grad_norm +1e-7) 
            e_r =  (grad)* scale
            self.vaccine_state["gradient"][module] = e_r.detach().clone()
            # print(module)
        #     print( torch.norm(self.vaccine_state["e_r"][module]) )
        # print(len(self.vaccine_state["e_r"]))
    
    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.vaccine_state["e_r"]:
        #     module.weight.data -= self.vaccine_state["e_r"][module]
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)



    @torch.no_grad()
    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([

                    ( poison_grads_representation[name] ).norm(p=2)
      
                    # ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for name in poison_grads_representation
                ]),
                p=2
               )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm




class RandomVaccineTrainer(Trainer):
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
                # print("gere2")
            return loss 

        self.vaccine_state = {}
        self.vaccine_state ["hooks"] = []
        self.vaccine_state ["gradient"] = {}
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)
        # for param in model.parameters():
        #     if param.grad is not None:
        #         param.grad*= 1/2
        
        # else:
        #     loss = step()
        return loss.detach() / self.args.gradient_accumulation_steps

            
    
    
    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            # print(perturbation[0,1,:])
            # # print(output.shape)
            # print(output[0,1,:])
            variance = self.args.rho
            # Generate samples from a Gaussian distribution
            gaussian_samples =  variance**(1/2) * torch.randn_like(output[0] )
            output[0].data =output[0] + gaussian_samples
            # print(output.shape)
            return output
           
        
        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    
        
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # print(layer._get_name())
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.vaccine_state["hooks"])
        
    
    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.vaccine_state["e_r"]:
        #     module.weight.data -= self.vaccine_state["e_r"][module]
        for hook in self.vaccine_state["hooks"]:
            hook.remove()
        self.vaccine_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)



    @torch.no_grad()
    def _grad_norm(self,poison_grads_representation):
        norm = torch.norm(
                torch.stack([
                    ( poison_grads_representation[name] ).norm(p=2)
                    for name in poison_grads_representation
                ]),
                p=2
               )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm

class FITrainer(Trainer):
    
    def init(self, model ):
        self.initial_weights = {}
        for name, module in model.named_modules():
            if "lora" in name  and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear):
                self.initial_weights[module] = module.weight.data.detach().clone()
        self.round = 0
        
        
        
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)                
        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs) 
                
            reg = 0
            for name, module in model.named_modules():
                if "lora" in name and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear):
                    reg += self.args.lamb * torch.sum(self.fisher_vector[module]* torch.square(module.weight -self.initial_weights[module] ))
                    # reg += self.args.lamb * torch.sum(torch.square(module.weight -self.initial_weights[module] ))
            # print(reg)
            loss +=reg
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss 
        
        if self.round==0:
            self. fisher_vector = {module : 0  for name, module in model.named_modules() if "lora" in name  and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear)}
            eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
            for stepsize, old_inputs in enumerate(eval_dataloader):
                # Update the observed num examples
                # print(inputs)
                model.zero_grad()
                old_inputs = self._prepare_inputs(old_inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, old_inputs) 
                self.accelerator.backward(loss)
                for name, module in model.named_modules():
                    if "lora" in name  and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear):
                        self.fisher_vector[module] += torch.square(module.weight.grad.data.detach().clone())
                        # print(self.fisher_vector[module])
                print(loss)
                
        
        loss = step()
        # print( sum([torch.norm(self.vaccine_state ["gradient"][module]) for module in self.vaccine_state ["gradient"]  ]))
        # leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        # for module in leaf_modules_with_grad:
        #     # print(module.q_proj.lora_A["default"])
        #     module.weight.grad*= (1-self.masks[index])
        #     index+=1
        self.round+=1
        return loss.detach() / self.args.gradient_accumulation_steps
    
    
class KLTrainer(Trainer):
    
    def init(self, model ):
        import copy 
        self.teacher_model_w = copy.deepcopy(model.state_dict())
        self.round = 0
        
        
        
    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)             
        
           
        def step():
            temp = {name: copy.deepcopy(param) for name, param in model.named_parameters() if param.requires_grad}
            with torch.no_grad():
                model.load_state_dict(self.teacher_model_w)
                teacher_outputs = self.model(**inputs,
                return_dict=True,
                use_cache=False,
                )
                model.load_state_dict(temp, strict=False)
            student_ouput = model(**inputs,
            return_dict=True,
            use_cache=False,
            )
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs) 

            import torch.nn.functional as F
            # Compute KL divergence
            kl_loss = self.args.lamb *  torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(student_ouput[1], dim=1),
                             F.softmax(teacher_outputs[1].detach(), dim=1)) 
                    # reg += self.args.lamb * torch.sum(torch.square(module.weight -self.initial_weights[module] ))
            # kl_loss = torch.mean(student_ouput[1])
            print(kl_loss)
            loss +=kl_loss
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss 

            
        loss = step()
        self.round+=1
        return loss.detach() / self.args.gradient_accumulation_steps
