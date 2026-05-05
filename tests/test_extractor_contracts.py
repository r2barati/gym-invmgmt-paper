import pytest

from gym_invmgmt import CoreEnv
from gym_invmgmt.extractors import (
    BAMPNNPoolExtractor,
    GNNFeaturesExtractor,
    GNNFeaturesExtractorV3,
    GNNPoolingExtractor,
    ResidualGCNPoolExtractor,
    SharedMLPExtractor,
    TransformerFeaturesExtractor,
    TransformerTemporalExtractor,
    TransferPerEdgePolicy,
)
from gym_invmgmt.wrappers.domain_features import DomainFeatureWrapper
from gym_invmgmt.wrappers.graph_only_wrapper import GraphOnlyWrapper


def test_extractor_public_exports_are_available():
    assert SharedMLPExtractor is not None
    assert GNNFeaturesExtractor is not None
    assert GNNFeaturesExtractorV3 is not None
    assert GNNPoolingExtractor is not None
    assert BAMPNNPoolExtractor is not None
    assert ResidualGCNPoolExtractor is not None
    assert TransferPerEdgePolicy is not None
    assert TransformerFeaturesExtractor is not None
    assert TransformerTemporalExtractor is not None


def test_v1_gnn_extractor_accepts_v1_domain_features():
    env = DomainFeatureWrapper(CoreEnv(scenario="base"), enhanced=False)
    GNNFeaturesExtractor(env.observation_space, scenario="base")


def test_v1_gnn_extractor_rejects_v2_domain_features():
    env = DomainFeatureWrapper(CoreEnv(scenario="base"), enhanced=True, grouped=True)
    with pytest.raises(ValueError, match="requires DomainFeatureWrapper\\(enhanced=False\\)"):
        GNNFeaturesExtractor(env.observation_space, scenario="base")


def test_gnn_pooling_extractor_accepts_graph_only_v1_features():
    env = GraphOnlyWrapper(CoreEnv(scenario="base"))
    GNNPoolingExtractor(env.observation_space, scenario="base")


def test_gnn_pooling_extractor_rejects_v2_domain_features():
    env = DomainFeatureWrapper(CoreEnv(scenario="base"), enhanced=True, grouped=True)
    with pytest.raises(ValueError, match="V1 graph-only extractor"):
        GNNPoolingExtractor(env.observation_space, scenario="base")
