"""FORK(exo-10gbe-wsl): auto role assignment — a CUDA instance is auto-linked as
the prefill source for a same-model Metal (Mac) decode instance."""

from exo.master.main import compute_auto_disaggregation_links
from exo.shared.types.backends import Backend
from exo.shared.types.common import ModelId, NodeId
from exo.shared.types.instance_link import InstanceLink, InstanceLinkId
from exo.shared.types.worker.instances import Instance, InstanceId
from exo.shared.types.worker.runners import RunnerId
from exo.worker.tests.unittests.conftest import (
    get_mlx_ring_instance,
    get_pipeline_shard_metadata,
)

MODEL = ModelId("mlx-community/test-model")


def _instance(instance_id: InstanceId, node_id: NodeId) -> Instance:
    runner = RunnerId()
    return get_mlx_ring_instance(
        instance_id=instance_id,
        model_id=MODEL,
        node_to_runner={node_id: runner},
        runner_to_shard={runner: get_pipeline_shard_metadata(MODEL, device_rank=0)},
    )


def test_links_cuda_prefill_to_metal_decode() -> None:
    cuda_node, metal_node = NodeId(), NodeId()
    cuda_id, metal_id = InstanceId(), InstanceId()
    instances = {
        cuda_id: _instance(cuda_id, cuda_node),
        metal_id: _instance(metal_id, metal_node),
    }
    node_backends = {
        cuda_node: [Backend.MlxCpu, Backend.MlxCuda],
        metal_node: [Backend.MlxCpu, Backend.MlxMetal],
    }

    links = compute_auto_disaggregation_links(instances, node_backends, {})

    assert len(links) == 1
    assert links[0].prefill_instances == [cuda_id]
    assert links[0].decode_instances == [metal_id]


def test_no_link_without_a_metal_decode() -> None:
    cuda_node = NodeId()
    cuda_id = InstanceId()
    instances = {cuda_id: _instance(cuda_id, cuda_node)}
    node_backends = {cuda_node: [Backend.MlxCpu, Backend.MlxCuda]}

    assert compute_auto_disaggregation_links(instances, node_backends, {}) == []


def test_idempotent_when_already_linked() -> None:
    cuda_node, metal_node = NodeId(), NodeId()
    cuda_id, metal_id = InstanceId(), InstanceId()
    instances = {
        cuda_id: _instance(cuda_id, cuda_node),
        metal_id: _instance(metal_id, metal_node),
    }
    node_backends = {
        cuda_node: [Backend.MlxCuda],
        metal_node: [Backend.MlxMetal],
    }
    existing = {
        InstanceLinkId(): InstanceLink(
            link_id=InstanceLinkId(),
            prefill_instances=[cuda_id],
            decode_instances=[metal_id],
        )
    }

    assert compute_auto_disaggregation_links(instances, node_backends, existing) == []
