# Aegis Health Plan: Claims Payment Policy Manual

> Synthetic policy written for this demo. It is modeled on common commercial-payer concepts but is not
> any real payer's policy and is not clinical guidance.

## AEG-1 Administrative requirements

### AEG-1.1 Timely filing
Claims must be received within 90 calendar days of the date of service. Claims received after the filing
limit are denied. Late filing cannot be overridden by clinical documentation.

### AEG-1.2 Member eligibility
Services are covered only when the member's coverage is active on the date of service. Services rendered
before the coverage start date or after the termination date are denied.

### AEG-1.3 Duplicate claims
A claim line that matches a previously processed line for the same member, rendering provider, date of
service and procedure code is a duplicate and is denied.

### AEG-1.4 Provider identifiers
The billing provider must have a valid National Provider Identifier (NPI). NPIs failing the check-digit
validation are rejected.

## AEG-2 Network requirements

### AEG-2.1 HMO network and referrals
Members enrolled in an HMO plan must receive non-emergency care from in-network providers, or from an
out-of-network provider with an approved referral. Non-emergency out-of-network services without a referral
are denied.

### AEG-2.2 Emergency services
Emergency services (place of service 23, emergency room) are covered regardless of network status.

## AEG-3 Prior authorization

### AEG-3.1 Services requiring prior authorization
Advanced imaging (MRI), arthroscopic surgery and chemotherapy administration require prior authorization.
An authorization must match the member and procedure code and cover the date of service. Non-emergency
services without a valid authorization are denied.

### AEG-3.2 Retrospective authorization for emergencies
When a service requiring authorization is performed in the emergency setting (place of service 23), the
absence of a prior authorization does not by itself result in denial. The service is payable when the
clinical record documents an emergency condition, such as sudden severe symptoms, acute neurological change,
trauma, or a time-sensitive diagnosis where delay could cause serious harm.

## AEG-4 Medical necessity

### AEG-4.1 General principle
A procedure is medically necessary when a diagnosis on the claim supports it. When no listed diagnosis
supports the procedure, the claim may still be paid if the clinical documentation meets the procedure-specific
criteria below. Documentation must state findings affirmatively; the absence or denial of a finding does
not satisfy a criterion.

### AEG-4.2 MRI of the brain (CPT 70553)
Covered diagnoses include seizure, multiple sclerosis, transient ischemic attack and brain neoplasm.
For headache, MRI of the brain is medically necessary only when at least one red flag is documented:
- sudden onset severe ("thunderclap") headache
- focal neurological deficit on examination
- papilledema
- new headache in a patient with cancer or immunosuppression
- headache that is progressively worsening over weeks
Chronic, stable headache with a normal neurological examination does not meet criteria.

### AEG-4.3 MRI of the lumbar spine (CPT 72148)
Covered diagnoses include lumbar disc displacement with radiculopathy. For low back pain, MRI is medically
necessary only when:
- at least six weeks of conservative therapy (physical therapy, medication) have failed, or
- red flags are present: progressive motor weakness, bowel or bladder dysfunction (cauda equina symptoms),
  suspected infection, or history of cancer.
Low back pain of less than six weeks without red flags does not meet criteria.

### AEG-4.4 Knee arthroscopy with meniscectomy (CPT 29881)
Medically necessary for a meniscal tear confirmed by imaging with mechanical symptoms (locking, catching)
and failure of at least six weeks of conservative management.

### AEG-4.5 Therapeutic procedures (CPT 97110, 97140, 97530)
Physical therapy is medically necessary for documented functional deficits with measurable goals.

## AEG-5 Coding

### AEG-5.1 Procedure-to-procedure (bundling) edits
Some procedure pairs are mutually exclusive or one is a component of the other when performed at the same
encounter. The component (column 2) code is denied. Where the edit allows a modifier, modifier 59 or XS may
bypass the edit only when the documentation shows a distinct procedural service: a different anatomic site,
a separate incision or lesion, or a separate session. Modifier use that is not supported by documentation
does not bypass the edit.

### AEG-5.2 Medically unlikely edits (units)
Units of service above the per-day maximum for a code are not payable. Units up to the maximum are paid.

### AEG-5.3 Evaluation and management level
CPT 99215 requires medical decision making of high complexity, for example: a decision regarding
hospitalization, drug therapy requiring intensive monitoring for toxicity, or a severe exacerbation of a
chronic illness. A routine follow-up of a stable condition does not support 99215.

### AEG-5.4 Charge outliers
A line billed at more than three times the reference cost per unit is reviewed. It is payable when the
documentation explains the additional work, for example a bilateral procedure, an unusually prolonged
service, or significant complications.

## AEG-6 Age-specific services

### AEG-6.1 Age limits
Some services are defined for specific ages (for example, the Medicare annual wellness visit G0439 for
members 65 and older; infant preventive visits 99391 for members under 1 year). Services billed outside the
defined age range are not payable as billed.
