from torch import nn
import torch

def object_is_basemodel(cls):
    for bc in cls.__bases__:
        if bc.__name__ in ['Module', 'BaseModelFSL', 'object', 'type']: 
            return bc.__name__ == 'BaseModelFSL'
        return object_is_basemodel(bc)

class BaseModelFSL(nn.Module):
    def __init__(self):
        super(BaseModelFSL, self).__init__()
        self.exp_indices = {}
        self.output = {}
        
    def set_meter(self, name: str, value):
        """
        Set experiment meters.
        Args:
            name (str): _description_
            value (Tensor or other type): _description_
        """
        self.exp_indices[name] = value
    
    def register_output(self, name, value):
        self.output[name] = value
    
    def get_meter(self, name: str):
        """
        Args:
            name (str): key

        Raises:
            KeyError: _description_
        Returns:
            _type_: _description_
        """
        if name not in self.exp_indices:
            raise KeyError("No such meter: '{}' in the meter set".format(name))
        return self.exp_indices[name]
    
    def set_output(self, name, value):
        self.output[name] = value
        self.exp_indices[name] = value.clone().detach()
    
    def register_sub_modules(self):
        """
            Check out all of attributes if it's 'BaseModelFSL' class, 
                collect its 'exp_indices'
        """
        submodules = [name for name, _ in self.named_children()]
        # print(self.__getattribute__('memory_block'))
        # exit()
        for submodl_name in submodules:
            submodl = getattr(self, submodl_name)
            if submodl.__class__.__name__ != 'function' and \
                object_is_basemodel(submodl.__class__) and submodl.exp_indices:
                meter_names = [k for k in submodl.exp_indices]
                output_names = [k for k in submodl.output]
                
                for meter_n in meter_names:
                    self.set_meter(meter_n, submodl.exp_indices.pop(meter_n))
                for output_n in output_names:
                    self.set_output(output_n, submodl.output.pop(output_n))

        
    def forward(self, *args, **kwargs) -> dict:
        """_summary_
        """
        self.exp_indices = {}
        outp_put = self.forward_(*args, **kwargs)
        outputs = {on: self.output[on] for on in self.output if on not in outp_put}
        self.register_sub_modules()
        return {**outp_put, **outputs}
    
    def forward_(self, *args, **kwargs) -> dict:
        raise NotImplementedError


class SimpleLinearModel(BaseModelFSL):
    def __init__(self, inp_dim):
        super(SimpleLinearModel, self).__init__()
        self.project = nn.Linear(inp_dim, 1)
    
    def forward_(self, x):
        y = self.project(x)
        self.exp_indices['l2'] = y.pow(2).sum(-1).sqrt().mean()
        return {'logits': y}


if __name__ == '__main__':
    inp_dim = 640
    test_model = SimpleLinearModel(inp_dim)
    test_x = torch.randn([4, inp_dim])
    print(test_model(test_x))