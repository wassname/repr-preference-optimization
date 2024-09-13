import enum

from .dpo import DPOTrainingArguments
from .reprpo_svd import ReprPOSVDTrainingArguments
from .reprpo_hs import ReprPOHSTrainingArguments
from .reprpo_side import ReprPOSideInTrainingArguments
from .reprpo_side import ReprPOSideOutTrainingArguments
from .reprpo_side_hra import ReprPOSideInHRATrainingArguments, ReprPOSideOutHRATrainingArguments
from .reprpo_ortho import ReprPOOrthoTrainingArguments
from .reprpo_hra import ReprPOHRATrainingArguments

class Methods(enum.Enum):
    dpo = DPOTrainingArguments
    reprpo_svd = ReprPOSVDTrainingArguments
    reprpo_hs = ReprPOHSTrainingArguments
    reprpo_side = ReprPOSideInTrainingArguments
    reprpo_sideout = ReprPOSideOutTrainingArguments
    reprpo_side_hra = ReprPOSideInHRATrainingArguments
    reprpo_sideout_hra = ReprPOSideOutHRATrainingArguments
    reprpo_ortho = ReprPOOrthoTrainingArguments
    reprpo_hrank = ReprPOHRATrainingArguments
