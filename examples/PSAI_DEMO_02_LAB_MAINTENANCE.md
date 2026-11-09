# Project Proposal: Laboratory Equipment Maintenance and Traceability Pilot

**Project ID:** LAB-MAINT-PILOT-2027  
**Document type:** Pilot scope, requirements, work plan, and acceptance criteria  
**Version:** 1.0 — proposed for coordinator review  
**Scenario period:** 2027-02-01 to 2027-03-19  
**Document status:** Synthetic demonstration case for a PSAI committee presentation. The laboratory, inventory, staffing, dates, costs, records, and targets are fictional assumptions. This document does not report an actual deployment, completed implementation, or verified performance improvement.

## 1. Executive summary

A hypothetical university teaching laboratory maintains 60 instruments across two rooms, including oscilloscopes, power supplies, and signal generators. Fault reports are recorded in spreadsheets and paper notebooks. A device may be marked available before its inspection has been documented, and different spreadsheets sometimes use different identifiers for the same instrument. The pilot proposes a local maintenance register, approval-aware status workflow, import validation, audit trail, and backup procedure.

The scope covers maintenance administration and evidence traceability. The application does not control instruments remotely, certify calibration, approve purchases, or replace the laboratory coordinator's responsibility for equipment safety.

## 2. Objectives and intended outcomes

The primary objective is to provide teaching staff with one consistent view of equipment availability and outstanding maintenance requests before practical classes.

The project aims to preserve a traceable sequence of fault reporting, inspection, authorization, and return to service for each registered instrument.

The project seeks to make duplicate inventory entries and unauthorized status changes visible before those errors affect teaching operations. A proposed evaluation target is that all status changes in the synthetic test set can be traced to an actor and timestamp; this target is not yet measured.

## 3. Current workflow and pilot boundaries

The scenario inventory contains 60 instruments: 35 in Room LAB-A and 25 in Room LAB-B. Existing records include equipment tags, manufacturer models, fault descriptions, and handwritten inspection notes, but they do not use a consistent format for maintenance state.

| Room | Scenario instruments | Typical record issues |
|---|---:|---|
| LAB-A | 35 | Duplicate equipment tags, inconsistent room codes |
| LAB-B | 25 | Missing serial numbers, incomplete inspection notes |
| **Total** | **60** | **Inventory and maintenance pilot** |

A user may report a fault, but only authorized maintenance staff or the coordinator may approve returning a device to service. Equipment with an unresolved safety-related fault must remain unavailable even when the assigned technician reports that an inspection is complete.

The pilot uses synthetic CSV records and attachments identified by references rather than storing raw student information. No real user credentials, student personal data, maintenance certificates, or confidential university records will be used in the demonstration.

### Explicit exclusions

Automatic instrument control, purchasing, vendor repair contracts, calibration certification, integration with university identity services, and automatic approval of equipment safety are excluded from this pilot. Importing historical paper notes by OCR is a possible later phase, not a committed delivery.

## 4. Planned implementation activities

We will review the 60 scenario inventory records, available field definitions, and representative maintenance forms with the laboratory coordinator.

We will design the equipment schema, request schema, audit-event structure, and permitted state transitions, including safety-hold approval rules.

We will implement a Python API backed by SQLite for equipment records, maintenance requests, inspection notes, and immutable audit entries.

We will build a request workflow that applies role permissions and prevents unauthorized transitions from `unavailable_safety_hold` to `available`.

We will implement CSV import preview, field mapping, duplicate-ID validation, and an all-or-nothing commit for accepted inventory updates.

We will create a dashboard with equipment availability, request triage, pending approvals, and room-level filters.

We will develop an audit history view that lets the coordinator inspect the actor, timestamp, prior state, and justification behind each change.

We will create a local backup and restore utility with a verification summary covering inventory, requests, and audit events.

We will test the workflow with synthetic normal, duplicate, incomplete, conflicting, and unauthorized cases, including a safety-hold release attempt.

We will deploy the pilot application in Docker on the laboratory demonstration computer and document installation, recovery, rollback, and operator training procedures.

## 5. Requirements

### 5.1 Equipment registry and import

The registry must store a unique equipment ID, room, model, serial number when available, service state, and maintenance history. A serial number must not be treated as a mandatory unique identifier when the scenario source lacks one.

The application must support CSV import with preview, field mapping, row-level validation, and explicit duplicate equipment-ID detection. A rejected import must not partially overwrite accepted inventory records.

A missing serial number should be recorded as unknown, not replaced with an invented value. Conflicting room assignments must be flagged for coordinator review before the proposed import is committed.

### 5.2 Maintenance states and permissions

The workflow must distinguish `available`, `fault_reported`, `awaiting_inspection`, `under_repair`, `awaiting_approval`, and `unavailable_safety_hold` states. A device on `unavailable_safety_hold` must not be returned to `available` without a documented authorization decision.

Teaching staff may create and view their own requests, while maintenance staff may record inspection findings and propose a return to service. Only the coordinator may approve closing a safety-related hold; maintenance staff may close ordinary requests only under the approved permission matrix.

Every status transition must preserve the prior state, new state, actor ID, UTC timestamp, justification, and associated request ID. Edits to a request description must not delete earlier audit events.

### 5.3 Requests, dashboard, and resilience

A request must capture the reporter role, equipment ID, fault summary, reported date, priority, and optional attachment reference. Priority labels are `low`, `normal`, and `urgent`, but the source does not authorize an automatic priority decision from free text.

The dashboard must show open requests, unavailable equipment, aging requests, and pending approvals, with filters by room, state, priority, and technician assignment.

The application must run on a laboratory-controlled local computer using a Python API and SQLite for the pilot. The system must remain usable without a public cloud service and must protect configuration secrets from export files.

The backup procedure must allow restoration of registry, request, and audit records. The system should make clear when a restored database is older than the latest exported activity log.

## 6. Deliverables and proposed acceptance checks

The final deliverables are the pilot API, SQLite schema, import-preview interface, maintenance dashboard, audit history view, backup utility, and operator handbook.

The handover report must distinguish completed checks from checks not executed, list observed failures, and identify unresolved policy questions. A proposed acceptance criterion must not be reported as a successful test result.

Acceptance case A1 requires a duplicate equipment-ID import to be rejected without modifying existing accepted records. Acceptance case A2 requires a record with an unknown serial number to remain marked unknown rather than being fabricated.

Acceptance case A3 requires a teaching-staff user to be blocked from releasing an instrument from safety hold. Acceptance case A4 requires a coordinator-approved return to service to retain the prior state, new state, actor, timestamp, and justification in audit history.

Acceptance case A5 requires restoring a backup into a clean test directory and comparing inventory, request, and audit-record counts and identifiers with the expected source snapshot. Acceptance case A6 requires the dashboard to distinguish equipment unavailable for inspection from equipment awaiting administrative approval.

These acceptance checks describe future work only. This proposal contains no test execution log, no pass rate, and no evidence that any case has passed.

## 7. Team, budget, and milestones

The scenario staffing plan includes two software developers, one maintenance representative, and one laboratory coordinator with final safety authority. Named owners, task assignments, and staff availability are not yet confirmed.

The proposed spending ceiling is 32000000 VND for the pilot, including setup, testing, and training materials. The budget is an illustrative planning constraint rather than an approved expenditure; no per-task allocation is supplied.

Milestone M1 is a schema and permission-matrix review proposed for 2027-02-09. Milestone M2 is a supervised workflow rehearsal proposed for 2027-03-05. Milestone M3 is a handover decision proposed for 2027-03-19.

Review of the inventory source must precede final field mapping. Schema and permission review must precede implementation of the release workflow. Testing requires the API, importer, audit history, and dashboard to be available. These are candidate dependencies rather than an approved critical path.

Task-level duration estimates, individual start and finish dates, training-session attendance, and recovery time objectives remain unspecified and require coordinator review.

## 8. Risks, unresolved decisions, and change control

Risk R1 is accidentally returning unsafe equipment to service. The proposed control is coordinator-only safety-hold release with a retained audit event and an explicit denial case in testing.

Risk R2 is overwriting valid records during CSV import. The proposed control is preview, duplicate detection, row-level validation, and transaction rollback on failed imports.

Risk R3 is losing the history needed to investigate a disputed change after database recovery. The proposed control is backup verification with audit-record reconciliation and an explicit notice of snapshot age.

Open decisions include the exact priority-escalation policy, whether ordinary repairs need coordinator sign-off, attachment retention, named role assignments, and the approved recovery objective. Changes to these decisions require a recorded coordinator review rather than an invented default.

## 9. Expected results and limits of interpretation

The expected result is a traceable view of unavailable instruments, maintenance requests, and authorization decisions for the two pilot rooms.

Teaching staff should be able to identify equipment unsuitable for a practical class, while the coordinator should be able to reconstruct why a status was changed and who authorized it.

Any statement that the pilot reduces preparation time, prevents equipment damage, or lowers maintenance costs requires a later comparison against real operating records. No such outcome is established by this synthetic proposal, and this document is not an independent validation of PSAI.
