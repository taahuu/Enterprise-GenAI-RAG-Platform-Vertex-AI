# Northwind Enterprises – Incident Response Runbook

## Severity levels
- SEV1: complete outage or data loss affecting many customers. Respond within 15 minutes.
- SEV2: major feature degraded for a large share of customers. Respond within 30 minutes.
- SEV3: minor degradation with a workaround available. Respond within 4 hours.
- SEV4: cosmetic issue or request. Handled during normal business hours.

## Roles and communication
The on-call engineer acts as incident commander until a more senior engineer takes over. All SEV1 and SEV2 incidents are coordinated in the #incident-war-room Slack channel. The communications lead posts a status page update every 30 minutes until the incident is resolved.

## After the incident
A blameless postmortem must be published within 5 business days of resolving any SEV1 or SEV2 incident. The postmortem lists the timeline, root cause, customer impact and follow-up actions with named owners.

## Escalation
If the on-call engineer does not acknowledge a page within 10 minutes, the page automatically escalates to the secondary on-call, then to the engineering manager.
