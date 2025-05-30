import torch
import torch.nn as nn
import torch.nn.functional as F

class NetworkMasker(nn.Module):
    def __init__(self, original_net: nn.Module, mask_logits=None, mask_type=None):
        """
        Wraps an existing network so that, during the forward pass,
        each activation layer has a 3-way categorical mask (active, deactivate, program).
        """
        super().__init__()
        self.network = original_net

        # discover which layers are maskable and how many channels/features each has
        maskable = NetworkMasker.maskable_layers(original_net)

        # build a ParameterDict of shape (3, C) logits for each layer
        if mask_logits is None:
            if mask_type == "network":
                self.mask_logits = nn.ParameterDict({
                    name: nn.Parameter(
                        torch.zeros(3, maskable[name]['size']),
                        requires_grad=True
                    )
                    for name in maskable if name != "input"
                })
            elif mask_type == "input":
                self.mask_logits = nn.ParameterDict({
                    name: nn.Parameter(
                        torch.zeros(3, maskable[name]['size']),
                        requires_grad=True
                    )
                    for name in maskable if name == "input"
                })
            elif mask_type == "both":
                self.mask_logits = nn.ParameterDict({
                    name: nn.Parameter(
                        torch.zeros(3, maskable[name]['size']),
                        requires_grad=True
                    )
                    for name in maskable
                })
            else:
                raise ValueError(f"Unknown mask_type: {mask_type}. Use 'network', 'input', or 'both'.")
        else:
            # testing: wrap given logits into a frozen ParameterDict
            if isinstance(mask_logits, nn.ParameterDict):
                # just turn off grads
                for p in mask_logits.values():
                    p.requires_grad = False
                self.mask_logits = mask_logits
            else:
                # assume it's a plain dict[name->Tensor]
                self.mask_logits = nn.ParameterDict({
                    name: nn.Parameter(mask_logits[name], requires_grad=False)
                    for name in mask_logits
                })

        # freeze the original network’s weights
        for p in self.network.parameters():
            p.requires_grad = False

    def forward(self, x):
        '''
        1: active
        -1: deactive
        0: part of the program
        '''
        
        # Assuming inputs are binary ---- Mask the *input* first ----
        if "input" in self.mask_logits:
            logits = self.mask_logits["input"]     # (3, C_input)
            probs = F.softmax(logits, dim=0)       # (3, C_input)
            p_act, p_deact, p_prog = probs.unbind(0)

            # broadcast shape = [1, C_input, 1, 1, ...] matching x
            shape = [1, -1] + [1] * (x.dim() - 2)
            p_act   = p_act.view(*shape).expand_as(x)
            p_deact = p_deact.view(*shape).expand_as(x)
            p_prog  = p_prog.view(*shape).expand_as(x)

            # define the three input‐branches
            active     = torch.ones_like(x)   # force 1
            deactivate = torch.zeros_like(x)  # force 0
            program    = x                     # keep original

            # soft blend
            x = p_act*active + p_deact*deactivate + p_prog*program
        
        assert isinstance(self.network, nn.Sequential), "NetworkMasker only supports nn.Sequential"
        # Assume the network is a Sequential. Mask the activations
        for name, module in self.network.named_children():
            out = module(x)
            if name in self.mask_logits:
                # 1) Grab the logits and convert to probabilities
                #    mask_logits[name]: shape = (3, C)
                logits = self.mask_logits[name]               # (3, C)
                probs = F.softmax(logits, dim=0)              # (3, C)
                p_act, p_deact, p_prog = probs.unbind(0)      # each is (C,)

                # 2) Broadcast each prob vector to out’s full shape
                #    e.g. out.shape = (B, C, H, W) or (B, C)
                shape = [1, -1] + [1] * (out.dim() - 2)

                p_act   = p_act.view(*shape).expand_as(out)
                p_deact = p_deact.view(*shape).expand_as(out)
                p_prog  = p_prog.view(*shape).expand_as(out)
                
                # 3) Build the three branches exactly as you did before
                if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
                    active     = x.clone()             # pass-through
                    deactivate = torch.zeros_like(out) # mask off
                    program    = out                   # normal activation
                elif isinstance(module, nn.Sigmoid):
                    active     = torch.ones_like(out)  # force 1
                    deactivate = torch.zeros_like(out) # force 0
                    program    = out
                elif isinstance(module, nn.Tanh):
                    active     = torch.ones_like(out)  # force +1
                    deactivate = -torch.ones_like(out) # force –1
                    program    = out
                else:
                    raise ValueError(f"Unsupported activation: {module}")

                # 4) Soft-blend all three
                out = p_act * active + p_deact * deactivate + p_prog * program
            # advance
            x = out
        return x

    @staticmethod
    def maskable_layers(network):
        """
        Returns a dictionary mapping incremental keys (as strings) to maskable activation layers 
        and their output sizes.

        Only activation layers (ReLU, LeakyReLU, Sigmoid, Tanh) are considered maskable.
        The output size is inferred from the most recent layer with a defined output size 
        (e.g. from a Linear or Conv2d layer via the attributes 'out_features' or 'out_channels').

        The returned dictionary maps keys (e.g. "0", "1", ...) to dictionaries with:
            - 'layer': the activation layer module,
            - 'size': the inferred output size for that activation.

        Returns:
            dict: A mapping from incremental string keys to dictionaries with keys 'layer' and 'size'.
        """
        maskable_layers = {}
        

        # Retrieve the underlying sequential module.
        seq_net = network

        last_output_size = None

        # Mask for the input
        if hasattr(seq_net[0], 'in_features'):
            last_output_size = seq_net[0].in_features
        elif hasattr(seq_net[0], 'in_channels'):
            last_output_size = seq_net[0].in_channels
        maskable_layers['input'] = {
            'layer': None,
            'size': last_output_size
        }

        # Iterate through layers to detect activation functions.
        for idx, module in enumerate(seq_net):
            # Update the most recent output size if the module defines it.
            if hasattr(module, 'out_features'):
                last_output_size = module.out_features
            elif hasattr(module, 'out_channels'):
                last_output_size = module.out_channels

            # Only activation layers are maskable.
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid, nn.Tanh)):
                if last_output_size is None:
                    # If we haven't encountered a preceding layer with a size, skip this activation.
                    continue
                maskable_layers[str(idx)] = {
                    'layer': module,
                    'size': last_output_size
                }
        return maskable_layers
