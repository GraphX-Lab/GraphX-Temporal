from tgx.models.temporal_model import TemporalModel

from .tagt import TGAN

class TGATModel(TemporalModel):
    def __init__(self, *args, **kwargs):
        super(TGATModel, self).__init__(*args, **kwargs)
        self.model = TGAN(*args, **kwargs)


    def forward(self, batch):
        return super().forward(batch)
    
