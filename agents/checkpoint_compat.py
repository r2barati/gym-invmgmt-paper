"""Compatibility aliases for released SB3 checkpoints.

Some released checkpoints were serialized with earlier module paths such as
``src.models.gnn_extractor_v3.GNNFeaturesExtractorV3``. This module registers
lightweight aliases so those checkpoints load against the released package
namespace without modifying their archived metadata.
"""

import sys
import types

# Register the compatibility namespace used by serialized checkpoints.
if 'src' not in sys.modules:
    src = types.ModuleType('src')
    sys.modules['src'] = src
else:
    src = sys.modules['src']

if 'src.models' not in sys.modules:
    src_models = types.ModuleType('src.models')
    sys.modules['src.models'] = src_models
    src.models = src_models
else:
    src_models = sys.modules['src.models']

# GNN V3 extractor
from gym_invmgmt.extractors import gnn_extractor_v3
sys.modules['src.models.gnn_extractor_v3'] = gnn_extractor_v3
src_models.gnn_extractor_v3 = gnn_extractor_v3

# GNN V1 extractor
from gym_invmgmt.extractors import gnn_extractor
sys.modules['src.models.gnn_extractor'] = gnn_extractor
src_models.gnn_extractor = gnn_extractor

# Transformer extractor
from gym_invmgmt.extractors import transformer_extractor
sys.modules['src.models.transformer_extractor'] = transformer_extractor
src_models.transformer_extractor = transformer_extractor

# Temporal Transformer extractor
from gym_invmgmt.extractors import transformer_temporal_extractor
sys.modules['src.models.transformer_temporal_extractor'] = transformer_temporal_extractor
src_models.transformer_temporal_extractor = transformer_temporal_extractor

# Shared MLP extractor
from gym_invmgmt.extractors import shared_mlp as shared_mlp_extractor
sys.modules['src.models.shared_mlp_extractor'] = shared_mlp_extractor
src_models.shared_mlp_extractor = shared_mlp_extractor

# Residual GCN Pool extractor
try:
    from gym_invmgmt.extractors import residual_gcn_pool_extractor
    sys.modules['src.models.residual_gcn_pool_extractor'] = residual_gcn_pool_extractor
    src_models.residual_gcn_pool_extractor = residual_gcn_pool_extractor
except ImportError:
    pass  # Optional extractor is not required for every release artifact.

# Transfer per-edge policy extractor
try:
    from gym_invmgmt.extractors import transfer_per_edge_policy
    sys.modules['src.models.transfer_per_edge_policy'] = transfer_per_edge_policy
    src_models.transfer_per_edge_policy = transfer_per_edge_policy
except ImportError:
    pass
