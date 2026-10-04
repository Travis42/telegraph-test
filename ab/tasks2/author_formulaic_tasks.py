#!/usr/bin/env python3
"""One-shot authoring helper: writes the formulaic-register task files for
ab/tasks2/ (workspace/status-register content with recurring schema)."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

TASKS = [
    dict(
        id="for_01", register="formulaic", len_class="short",
        passage=(
            "STATUS host=api-01 role=web cpu=72% mem=61% disk=44% net=ok "
            "uptime=14d version=2.3.1 notes=none"
        ),
        question="What is the disk usage on api-01?",
        expected_answer="44%",
        facts=[
            "host is api-01", "role is web", "cpu is 72%",
            "mem is 61%", "disk is 44%", "net is ok",
            "uptime is 14d", "version is 2.3.1",
        ],
    ),
    dict(
        id="for_02", register="formulaic", len_class="short",
        passage=(
            "STATUS host=api-02 role=web cpu=91% mem=78% disk=44% net=degraded "
            "uptime=14d version=2.3.1 notes=latency-elevated-since-09:40"
        ),
        question="What is the cpu usage on api-02?",
        expected_answer="91%",
        facts=[
            "host is api-02", "role is web", "cpu is 91%",
            "mem is 78%", "disk is 44%", "net is degraded",
            "uptime is 14d", "version is 2.3.1",
            "notes say latency elevated since 09:40",
        ],
    ),
    dict(
        id="for_03", register="formulaic", len_class="short",
        passage=(
            "STATUS host=db-01 role=postgres cpu=38% mem=84% disk=71% net=ok "
            "uptime=203d version=15.4 notes=vacuum-running"
        ),
        question="What role does db-01 serve?",
        expected_answer="postgres",
        facts=[
            "host is db-01", "role is postgres", "cpu is 38%",
            "mem is 84%", "disk is 71%", "net is ok",
            "uptime is 203d", "version is 15.4", "notes say vacuum running",
        ],
    ),
    dict(
        id="for_04", register="formulaic", len_class="short",
        passage=(
            "STATUS host=db-02 role=postgres-replica cpu=35% mem=82% disk=70% "
            "net=ok uptime=203d version=15.4 notes=replication-lag=2s"
        ),
        question="What is the replication lag on db-02?",
        expected_answer="2s",
        facts=[
            "host is db-02", "role is postgres-replica", "cpu is 35%",
            "mem is 82%", "disk is 70%", "net is ok",
            "uptime is 203d", "version is 15.4", "replication lag is 2s",
        ],
    ),
    dict(
        id="for_05", register="formulaic", len_class="short",
        passage=(
            "STANDUP 2026-09-30 agent=checker squad=pipeline "
            "yesterday=reviewed-pr-4832,ran-batch-7 "
            "today=finish-batch-7-review,start-batch-8 "
            "blockers=none hours=6"
        ),
        question="What will checker work on after finishing the batch-7 review?",
        expected_answer="start batch 8",
        facts=[
            "standup date is 2026-09-30", "agent is checker",
            "squad is pipeline", "yesterday: reviewed pr-4832",
            "yesterday: ran batch-7", "today: finish batch-7 review",
            "today: start batch-8", "blockers: none", "hours: 6",
        ],
    ),
    dict(
        id="for_06", register="formulaic", len_class="short",
        passage=(
            "STANDUP 2026-09-30 agent=deployer squad=pipeline "
            "yesterday=shipped-v2.3.1-to-canary "
            "today=promote-canary-to-full-if-metrics-hold "
            "blockers=waiting-on-metrics-signoff hours=4"
        ),
        question="What is deployer blocked on?",
        expected_answer="waiting on metrics signoff",
        facts=[
            "standup date is 2026-09-30", "agent is deployer",
            "squad is pipeline", "yesterday: shipped v2.3.1 to canary",
            "today: promote canary to full if metrics hold",
            "blockers: waiting on metrics signoff", "hours: 4",
        ],
    ),
    dict(
        id="for_07", register="formulaic", len_class="short",
        passage=(
            "STANDUP 2026-09-29 agent=checker squad=pipeline "
            "yesterday=triaged-batch-6-flakes,fixed-docker-cache "
            "today=reviewed-pr-4832,ran-batch-7 blockers=none hours=7"
        ),
        question="How many hours did checker log on 2026-09-29?",
        expected_answer="7",
        facts=[
            "standup date is 2026-09-29", "agent is checker",
            "squad is pipeline", "yesterday: triaged batch-6 flakes",
            "yesterday: fixed docker cache", "today: reviewed pr-4832",
            "today: ran batch-7", "blockers: none", "hours: 7",
        ],
    ),
    dict(
        id="for_08", register="formulaic", len_class="short",
        passage=(
            "STANDUP 2026-09-30 agent=librarian squad=infra "
            "yesterday=reindexed-docs,migrated-3-wikis "
            "today=archive-stale-runbooks blockers=none hours=5"
        ),
        question="Which squad does librarian belong to?",
        expected_answer="infra",
        facts=[
            "standup date is 2026-09-30", "agent is librarian",
            "squad is infra", "yesterday: reindexed docs",
            "yesterday: migrated 3 wikis", "today: archive stale runbooks",
            "blockers: none", "hours: 5",
        ],
    ),
    dict(
        id="for_09", register="formulaic", len_class="short",
        passage=(
            "CHECKLIST name=pre-deploy v=7 items=6 "
            "[1]tests-green [2]migration-dry-run-ok [3]backup-taken "
            "[4]canary-5pct-15min [5]alerts-paged-on [6]rollback-doc-linked "
            "state=items-1-5-done,item-6-pending owner=deployer"
        ),
        question="Which checklist item is still pending?",
        expected_answer="item 6 (rollback doc linked)",
        facts=[
            "checklist name is pre-deploy", "checklist version is 7",
            "checklist has 6 items", "item 1 tests green is done",
            "item 2 migration dry-run ok is done", "item 3 backup taken is done",
            "item 4 canary 5pct 15min is done", "item 5 alerts paged on is done",
            "item 6 rollback doc linked is pending", "owner is deployer",
        ],
    ),
    dict(
        id="for_10", register="formulaic", len_class="short",
        passage=(
            "CHECKLIST name=nightly-verify v=3 items=5 "
            "[1]snapshot-exists [2]checksums-match [3]restore-test-ok "
            "[4]metrics-exported [5]report-posted state=all-done owner=librarian"
        ),
        question="How many items are in the nightly-verify checklist?",
        expected_answer="5",
        facts=[
            "checklist name is nightly-verify", "checklist version is 3",
            "checklist has 5 items", "item 1 snapshot exists is done",
            "item 2 checksums match is done", "item 3 restore test ok is done",
            "item 4 metrics exported is done", "item 5 report posted is done",
            "state is all done", "owner is librarian",
        ],
    ),
    dict(
        id="for_11", register="formulaic", len_class="short",
        passage=(
            "CHECKLIST name=incident-handover v=2 items=4 "
            "[1]timeline-written [2]sev-assigned [3]comms-sent "
            "[4]followups-ticketed state=items-1-3-done,item-4-pending "
            "owner=oncall"
        ),
        question="Who owns the incident-handover checklist?",
        expected_answer="oncall",
        facts=[
            "checklist name is incident-handover", "checklist version is 2",
            "checklist has 4 items", "item 1 timeline written is done",
            "item 2 sev assigned is done", "item 3 comms sent is done",
            "item 4 followups ticketed is pending", "owner is oncall",
        ],
    ),
    dict(
        id="for_12", register="formulaic", len_class="short",
        passage=(
            "CONFIG name=api-gateway env=prod replicas=4 "
            "timeout_ms=8000 retry=2 ratelimit_rps=5000 "
            "features=mtls:on,caching:on,canary:off owner=squad-pipeline"
        ),
        question="What is the request timeout in the api-gateway prod config?",
        expected_answer="8000 ms",
        facts=[
            "config name is api-gateway", "env is prod", "replicas is 4",
            "timeout is 8000 ms", "retry is 2", "ratelimit is 5000 rps",
            "feature mtls is on", "feature caching is on",
            "feature canary is off", "owner is squad-pipeline",
        ],
    ),
    dict(
        id="for_13", register="formulaic", len_class="short",
        passage=(
            "CONFIG name=api-gateway env=staging replicas=1 "
            "timeout_ms=8000 retry=2 ratelimit_rps=500 "
            "features=mtls:on,caching:off,canary:on owner=squad-pipeline"
        ),
        question="Which features differ between staging and prod in caching and canary terms (per this config)?",
        expected_answer="staging has caching off and canary on",
        facts=[
            "config name is api-gateway", "env is staging", "replicas is 1",
            "timeout is 8000 ms", "retry is 2", "ratelimit is 500 rps",
            "feature mtls is on", "feature caching is off",
            "feature canary is on", "owner is squad-pipeline",
        ],
    ),
    dict(
        id="for_14", register="formulaic", len_class="short",
        passage=(
            "CONFIG name=batch-runner env=prod replicas=2 "
            "timeout_ms=3600000 retry=0 ratelimit_rps=0 "
            "features=mtls:on,caching:off,canary:off owner=squad-pipeline "
            "schedule=cron-0-4-*.mon-fri"
        ),
        question="How many retries does batch-runner use?",
        expected_answer="0",
        facts=[
            "config name is batch-runner", "env is prod", "replicas is 2",
            "timeout is 3600000 ms", "retry is 0", "ratelimit is 0 rps",
            "feature mtls is on", "feature caching is off",
            "feature canary is off", "owner is squad-pipeline",
            "schedule is cron 0 4 * mon-fri",
        ],
    ),
]


def main():
    for task in TASKS:
        path = os.path.join(HERE, task["id"] + ".json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(task, fh, sort_keys=True, indent=2, ensure_ascii=False)
            fh.write("\n")
    print(f"{len(TASKS)} formulaic tasks written")


if __name__ == "__main__":
    main()
