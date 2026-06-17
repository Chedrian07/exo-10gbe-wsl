"""
Test that EventRouter immediately requests the next gap after draining a chunk
during cold-join catch-up, without waiting for another out-of-order live event.
"""

import anyio

from exo.routing.event_router import EventRouter
from exo.shared.types.commands import ForwarderCommand, RequestEventLog
from exo.shared.types.common import NodeId, SessionId
from exo.shared.types.events import GlobalForwarderEvent, LocalForwarderEvent, TestEvent
from exo.utils.channels import channel


def _make_session() -> SessionId:
    master_node_id = NodeId()
    return SessionId(master_node_id=master_node_id, election_clock=0)


def _make_global_event(session: SessionId, origin_idx: int) -> GlobalForwarderEvent:
    return GlobalForwarderEvent(
        origin_idx=origin_idx,
        origin=session.master_node_id,
        session=session,
        event=TestEvent(),
    )


async def test_catchup_requests_next_gap_after_draining():
    """
    Scenario:
    1. A live event arrives at idx=2000 — buffer holds it, gap at 0.
       Router issues RequestEventLog(since_idx=0).
    2. Master responds with chunk 0–999 (1 000 events).
       Router drains 0–999, but buffer still holds idx=2000.
       Gap now at idx=1000.
       Router must immediately issue RequestEventLog(since_idx=1000)
       WITHOUT another out-of-order event arriving.
    """
    session = _make_session()

    ext_in_send, ext_in_recv = channel[GlobalForwarderEvent]()
    cmd_send, cmd_recv = channel[ForwarderCommand]()
    local_out_send, _local_out_recv = channel[LocalForwarderEvent]()

    router = EventRouter(
        session_id=session,
        command_sender=cmd_send,
        external_inbound=ext_in_recv,
        external_outbound=local_out_send,
    )
    # Eliminate nack delay so the test completes in milliseconds.
    router._nack_base_seconds = 0.0  # pyright: ignore[reportPrivateUsage]

    collected_requests: list[RequestEventLog] = []

    async def collect_commands():
        with cmd_recv as commands:
            async for cmd in commands:
                if isinstance(cmd.command, RequestEventLog):
                    collected_requests.append(cmd.command)

    async def drive():
        async with anyio.create_task_group() as tg:
            tg.start_soon(router.run)
            tg.start_soon(collect_commands)

            # Step 1: live event at idx=2000 arrives (gap at 0).
            await ext_in_send.send(_make_global_event(session, origin_idx=2000))

            # Wait for the first RequestEventLog(since_idx=0).
            deadline = anyio.current_time() + 5.0
            while not collected_requests:
                await anyio.sleep(0.01)
                assert anyio.current_time() < deadline, (
                    "Timed out waiting for first RequestEventLog"
                )

            assert collected_requests[0].since_idx == 0, (
                f"Expected first request for idx=0, got {collected_requests[0].since_idx}"
            )

            # Step 2: master sends chunk 0–999.
            for idx in range(1000):
                await ext_in_send.send(_make_global_event(session, origin_idx=idx))

            # Wait for the follow-up RequestEventLog(since_idx=1000).
            deadline = anyio.current_time() + 5.0
            while len(collected_requests) < 2:
                await anyio.sleep(0.01)
                assert anyio.current_time() < deadline, (
                    "Timed out waiting for follow-up RequestEventLog after draining chunk"
                )

            assert collected_requests[1].since_idx == 1000, (
                f"Expected follow-up request for idx=1000, got {collected_requests[1].since_idx}"
            )

            ext_in_send.close()
            tg.cancel_scope.cancel()

    await drive()


async def test_no_spurious_request_when_fully_caught_up():
    """
    When the buffer is fully caught up (no forward gap), the router must NOT
    emit a spurious RequestEventLog.
    """
    session = _make_session()

    ext_in_send, ext_in_recv = channel[GlobalForwarderEvent]()
    cmd_send, cmd_recv = channel[ForwarderCommand]()
    local_out_send, _local_out_recv = channel[LocalForwarderEvent]()

    router = EventRouter(
        session_id=session,
        command_sender=cmd_send,
        external_inbound=ext_in_recv,
        external_outbound=local_out_send,
    )
    router._nack_base_seconds = 0.0  # pyright: ignore[reportPrivateUsage]

    collected_requests: list[RequestEventLog] = []

    async def collect_commands():
        with cmd_recv as commands:
            async for cmd in commands:
                if isinstance(cmd.command, RequestEventLog):
                    collected_requests.append(cmd.command)

    async def drive():
        async with anyio.create_task_group() as tg:
            tg.start_soon(router.run)
            tg.start_soon(collect_commands)

            # Send events 0–9 in order — no gap, fully sequential.
            for idx in range(10):
                await ext_in_send.send(_make_global_event(session, origin_idx=idx))

            # Give the router a moment to process.
            await anyio.sleep(0.05)

            # No RequestEventLog should have been issued.
            assert collected_requests == [], (
                f"Unexpected RequestEventLog(s) emitted for in-order delivery: {collected_requests}"
            )

            ext_in_send.close()
            tg.cancel_scope.cancel()

    await drive()
