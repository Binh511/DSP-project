# Project Proposal: Miner Fleet Telemetry and Incident Response Pilot

**Project ID:** MINER-OPS-PILOT-2027  
**Document type:** Technical project proposal and acceptance plan  
**Version:** 1.0 — proposed, not approved  
**Scenario period:** 2027-03-01 to 2027-04-16  
**Document status:** Synthetic demonstration case for a PSAI committee presentation. All organizations, assets, staffing, dates, costs, risks, and targets below are scenario assumptions, not evidence of an actual deployment. No activity or acceptance test is claimed to have been completed.

## 1. Executive summary

A hypothetical infrastructure operator manages 24 miner nodes across two facilities. Operators currently read process logs, check host status, and record incidents in separate spreadsheets. During a network interruption, a stale heartbeat can be mistaken for a failed node; repeated manual checks also make incident histories hard to reconstruct. The proposed pilot combines telemetry collection, a local dashboard, alert routing, and source-linked incident records without controlling miner workloads or accessing wallet credentials.

The pilot is limited to monitoring and operator review. It does not authorize a change to node configuration, automatic restart of services, mining optimization, or public Internet exposure of monitoring endpoints.

## 2. Objectives and measurable targets

The primary objective is to give operations staff a consistent view of node availability, telemetry freshness, and incident status across the two facilities.

The project aims to reduce the time needed to identify which node requires investigation. A target of identifying the affected node within five minutes of a simulated critical event is proposed for acceptance; this is not a measured improvement over the current process.

The objective is to make every operator acknowledgment and change in incident status traceable to an actor, a timestamp, and the related alert. A second proposed target is to preserve at least 95 percent of expected telemetry samples during a controlled, healthy-network rehearsal; no operational availability figure is claimed.

## 3. Scope and operating assumptions

The scenario inventory contains 24 miner nodes, grouped into 14 nodes at Facility A and 10 at Facility B. Each node has an asset identifier, a facility assignment, and an agreed contact point for operations.

| Facility | Scenario nodes | Local observation focus |
|---|---:|---|
| Facility A | 14 | Heartbeat freshness, process state, job throughput |
| Facility B | 10 | Disk usage, collector reachability, incident audit |
| **Total** | **24** | **Read-only telemetry pilot** |

Historical logs contain UTC timestamps, but local staff may enter incident notes in local time. The system must retain the original timestamp and a normalized UTC representation when the source permits conversion.

The pilot runs on an operator-controlled local server and uses synthetic or sanitized telemetry in demonstrations. Actual fleet credentials, wallet keys, staff passwords, and production incident logs are outside this demonstration. The operations lead is responsible for approving alert thresholds before they are used in a live environment.

### Exclusions

The pilot excludes autonomous remediation, miner process restarts, changes to wallet configuration, repair scheduling, service-level penalty calculations, and cross-company data sharing. Integrations with external messaging providers are candidates for later review, not committed scope.

## 4. Planned implementation activities

We will inventory the 24 scenario nodes and confirm facility mapping, ownership fields, and available telemetry sources with the operations lead.

We will design a normalized telemetry schema that separates raw source events, collector health, node observations, alert states, and operator audit events.

We will implement a Python collector that reads synthetic miner process logs and host metrics, attaches UTC timestamps, and preserves the source event identifier.

We will develop ingestion validation and idempotency checks so duplicate events, unknown nodes, malformed timestamps, and missing fields receive explicit outcomes.

We will build a local monitoring API for filtered node status, historical telemetry, incidents, and acknowledgment history.

We will create a dashboard that distinguishes `healthy`, `unreachable`, and `unknown` states and displays facility-level data freshness without suggesting automated recovery actions.

We will implement rule-based alert candidates for heartbeat age, disk usage, and collector failures, with threshold profiles marked as pending operator approval.

We will test ingestion, deduplication, role permissions, stale heartbeat detection, connectivity failures, and audit integrity using scripted fault-injection scenarios.

We will create an encrypted-backup rehearsal and a restore verification procedure that checks record counts, configuration integrity, and audit history after recovery.

We will deploy the pilot services with Docker on a local demonstration server and document installation, rollback, troubleshooting, and operator handover steps.

## 5. Functional and non-functional requirements

### 5.1 Telemetry and data integrity

The collector must record node identifier, observed timestamp, heartbeat age, process state, disk usage percentage, and job throughput when those fields are present in the source.

The system must distinguish a node marked `unreachable` from a collector that is unable to observe the node because of a facility-wide network disruption. Missing telemetry must be displayed as `unknown` until there is sufficient evidence to classify the incident.

Telemetry ingestion must be idempotent for a combination of node identifier, source event identifier, and source timestamp. Repeated uploads must not inflate event counts or create duplicate incidents.

The API must reject records with an unknown asset identifier and must record a reason for rejection without logging secret values. Raw evidence and normalized fields must remain linked for later review.

### 5.2 Operator access and audit

The monitoring application must distinguish viewer, operator, and administrator roles. Viewers may inspect dashboards; operators may acknowledge alerts; administrators may manage approved threshold profiles.

Every incident status change must retain the preceding status, the new status, actor, and UTC timestamp. The audit record must not be overwritten by subsequent edits to an incident description.

API credentials must be stored in local configuration or an approved secret store, not committed to source control or embedded in dashboard links. The pilot must remain usable without an external cloud service.

### 5.3 Dashboard, alerting, and reliability

The dashboard must show facility, node, telemetry freshness, active alerts, and last operator acknowledgment. Filters must allow operators to select a facility, severity, and node state.

A critical heartbeat alert should be proposed only after 180 seconds without a valid heartbeat when collector connectivity has been verified. A loss of collector connectivity must produce a separate data-quality warning rather than being reported as a confirmed node failure.

The collector must use bounded retries for transient failures and must preserve a local diagnostic record of exhausted retries. The pilot should keep a 30-day rolling retention window for normalized telemetry; final retention approval remains with the operations lead.

The recovery procedure must identify how to restore local configuration, telemetry index, and operator audit records from an encrypted backup. The precise recovery time objective is not yet approved.

## 6. Proposed outputs and acceptance criteria

The final deliverables are a Python collector, a local monitoring API, a read-only status dashboard, an alert-rule configuration, a restore procedure, and an operator guide.

The handover report must contain the configuration version, data sources used, test cases, observed outcomes, unresolved defects, and known assumptions. It must not present scenario targets as achieved results.

Proposed acceptance checks are: a repeated event does not produce a duplicate incident; an unknown node ID is rejected; a facility outage is shown as a data-quality problem rather than a confirmed node failure; a viewer cannot acknowledge an alert; and an authorized acknowledgment leaves an immutable audit entry.

A further acceptance check requires restoring a test backup and matching the expected inventory and audit records. The proposed five-minute identification target and 95-percent telemetry-sample target will be measured only in a future controlled rehearsal.

No acceptance check in this proposal is reported as passed. Test results must be recorded separately after execution and reviewed by the operations lead.

## 7. Resources, milestones, and constraints

The scenario team consists of two software engineers, one part-time site reliability engineer, and one operations lead responsible for decisions and sign-off. Availability and individual task ownership are not confirmed.

The proposed budget ceiling is 85000000 VND for this pilot, including local infrastructure and testing effort. This is a planning assumption, not an approved allocation; the document does not assign costs to individual activities.

Milestone M1 is a schema and evidence-source review proposed for 2027-03-09. Milestone M2 is an integrated rehearsal proposed for 2027-03-26. Milestone M3 is a handover review proposed for 2027-04-16.

Data-source inventory must precede final schema approval. The schema must be reviewed before ingestion behavior is frozen. Integrated alert tests depend on collector, ingestion, API, and dashboard components. These are proposed dependencies, not an approved project schedule.

Individual task start and end dates, effort estimates, on-call rotations, and threshold approvers remain unresolved. The operations lead must review these omissions before implementation commitments are made.

## 8. Risks, decisions, and open questions

Risk R1 is confusing missing data with confirmed node downtime. The mitigation is to separate collector health from node state and preserve the evidence behind each alert.

Risk R2 is duplicate incident creation after network recovery. The mitigation is idempotent ingestion and replay tests using repeated source events.

Risk R3 is exposing operational credentials in logs or dashboards. The mitigation is local secret handling, permission checks, and logging tests designed to detect leaked values.

A decision is pending on the final disk-usage threshold, approved retention period, backup encryption key custody, and responsibility for incident acknowledgment outside office hours. The proposal does not invent owners or approvals for these items.

## 9. Expected results and evaluation boundary

The expected result is a single, traceable monitoring view that helps staff identify which nodes need investigation and why an alert was raised.

Operators should be able to distinguish telemetry gaps from confirmed service failures, review the evidence behind an incident, and inspect the audit history of responses.

Any claim that the pilot reduces downtime, increases mining revenue, or improves mean time to resolution requires a future comparison against operational baseline data. This synthetic document does not provide that evidence and is not an independent evaluation of PSAI.
