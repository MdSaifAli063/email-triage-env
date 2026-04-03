"""
Email dataset for the Email Triage environment.
Each entry contains the email + ground-truth labels for all three tasks.
Tasks:
  Task 1 (easy)   — classify urgency + category
  Task 2 (medium) — classify + extract sender_name, deadline, sentiment
  Task 3 (hard)   — classify + extract + draft a contextually appropriate reply
"""

from typing import Any, Dict, List

EMAILS: List[Dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # EASY set — clear signals, unambiguous labels
    # -----------------------------------------------------------------------
    {
        "email": {
            "id": "e001",
            "subject": "URGENT: Production server down — customers can't log in",
            "sender": "ops-team@acme.com",
            "sender_domain": "acme.com",
            "body": (
                "Hi team,\n\n"
                "Our production server has been unresponsive for the last 20 minutes. "
                "Customers are unable to log in and we are losing revenue. "
                "Please escalate immediately.\n\n"
                "Best,\nMarcus Lee\nHead of Operations"
            ),
            "received_at": "2024-06-15T09:03:00Z",
            "has_attachment": False,
            "thread_length": 1,
        },
        "ground_truth": {
            "urgency": "urgent",
            "category": "technical",
            "sender_name": "Marcus Lee",
            "deadline": None,
            "sentiment": "angry",
            "reply_must_contain": ["investigating", "escalat", "team"],
            "reply_tone": "apologetic_and_action_oriented",
        },
    },
    {
        "email": {
            "id": "e002",
            "subject": "Congratulations! You've won a $500 Amazon gift card",
            "sender": "noreply@prize-winner99.net",
            "sender_domain": "prize-winner99.net",
            "body": (
                "Dear valued customer,\n\n"
                "You have been selected as our lucky winner. "
                "Click here to claim your $500 Amazon gift card now: http://scam-link.xyz\n\n"
                "Hurry, offer expires in 24 hours!"
            ),
            "received_at": "2024-06-15T10:15:00Z",
            "has_attachment": False,
            "thread_length": 1,
        },
        "ground_truth": {
            "urgency": "spam",
            "category": "spam",
            "sender_name": None,
            "deadline": None,
            "sentiment": "neutral",
            "reply_must_contain": [],
            "reply_tone": "none",
        },
    },
    {
        "email": {
            "id": "e003",
            "subject": "Invoice #INV-2024-0891 — Payment due",
            "sender": "billing@vendor-corp.com",
            "sender_domain": "vendor-corp.com",
            "body": (
                "Dear Accounts Payable,\n\n"
                "Please find attached invoice #INV-2024-0891 for $12,450.00 "
                "due on 2024-06-30. Kindly process payment before the due date "
                "to avoid a late fee.\n\n"
                "Regards,\nSophia Chen\nBilling Department, Vendor Corp"
            ),
            "received_at": "2024-06-14T14:22:00Z",
            "has_attachment": True,
            "thread_length": 1,
        },
        "ground_truth": {
            "urgency": "normal",
            "category": "billing",
            "sender_name": "Sophia Chen",
            "deadline": "2024-06-30",
            "sentiment": "neutral",
            "reply_must_contain": ["received", "process", "payment"],
            "reply_tone": "professional_acknowledgment",
        },
    },
    # -----------------------------------------------------------------------
    # MEDIUM set — mixed signals, extraction required
    # -----------------------------------------------------------------------
    {
        "email": {
            "id": "e004",
            "subject": "Follow-up: Contract renewal discussion",
            "sender": "james.whitfield@bigclient.io",
            "sender_domain": "bigclient.io",
            "body": (
                "Hello,\n\n"
                "I wanted to follow up on our conversation last week about renewing "
                "the enterprise contract. We need a revised proposal by July 15th at "
                "the latest — our board meets on the 18th and they'll need time to review.\n\n"
                "Also, could you loop in your legal team? There are a few clauses we'd "
                "like to negotiate.\n\n"
                "Best regards,\nJames Whitfield\nVP Procurement, BigClient Inc."
            ),
            "received_at": "2024-06-13T11:45:00Z",
            "has_attachment": False,
            "thread_length": 3,
        },
        "ground_truth": {
            "urgency": "urgent",
            "category": "sales",
            "sender_name": "James Whitfield",
            "deadline": "2024-07-15",
            "sentiment": "neutral",
            "reply_must_contain": ["proposal", "legal", "July"],
            "reply_tone": "professional_and_responsive",
        },
    },
    {
        "email": {
            "id": "e005",
            "subject": "Employee complaint — hostile work environment",
            "sender": "anonymous@safereport.hr",
            "sender_domain": "safereport.hr",
            "body": (
                "This is a confidential report submitted through the anonymous HR portal.\n\n"
                "An employee in the Engineering department has been subjected to repeated "
                "derogatory comments from their direct manager over the past two months. "
                "The behavior constitutes harassment under company policy section 4.2. "
                "The employee fears retaliation and requests immediate intervention.\n\n"
                "Please treat this report with the highest confidentiality."
            ),
            "received_at": "2024-06-12T08:30:00Z",
            "has_attachment": False,
            "thread_length": 1,
        },
        "ground_truth": {
            "urgency": "urgent",
            "category": "hr",
            "sender_name": None,
            "deadline": None,
            "sentiment": "negative",
            "reply_must_contain": ["confidential", "investigate", "HR"],
            "reply_tone": "empathetic_and_formal",
        },
    },
    {
        "email": {
            "id": "e006",
            "subject": "Office supplies order — Q3 request",
            "sender": "admin@acme.com",
            "sender_domain": "acme.com",
            "body": (
                "Hi,\n\n"
                "Attached is the Q3 office supplies request from all departments. "
                "Total estimated spend is $840. Please approve so we can place the order "
                "before the end of the month.\n\n"
                "No rush on this one — whenever you get a chance.\n\n"
                "Thanks,\nPriya Nair\nOffice Manager"
            ),
            "received_at": "2024-06-11T16:00:00Z",
            "has_attachment": True,
            "thread_length": 1,
        },
        "ground_truth": {
            "urgency": "low",
            "category": "general",
            "sender_name": "Priya Nair",
            "deadline": None,
            "sentiment": "positive",
            "reply_must_contain": ["approved", "order", "supplies"],
            "reply_tone": "brief_approval",
        },
    },
    # -----------------------------------------------------------------------
    # HARD set — ambiguous, nuanced, complex reply required
    # -----------------------------------------------------------------------
    {
        "email": {
            "id": "e007",
            "subject": "Re: Re: Re: Data breach notification — legal hold",
            "sender": "counsel@legalfirm-partners.com",
            "sender_domain": "legalfirm-partners.com",
            "body": (
                "Dear Ms. Rivera,\n\n"
                "Further to our call this morning, I must emphasize the gravity of the "
                "situation. The data breach affecting approximately 14,000 customer records "
                "triggers mandatory notification obligations under GDPR Article 33 and "
                "CCPA §1798.82. Notification to regulators must be completed within 72 hours "
                "of discovery — which was yesterday at 3 PM. You are now at T+18 hours.\n\n"
                "We strongly advise you to:\n"
                "1. Immediately engage your DPO to draft regulator notification\n"
                "2. Preserve all system logs under legal hold\n"
                "3. Do NOT communicate details externally until we clear messaging\n\n"
                "We need a call with your CTO and CISO by end of day today. "
                "Please confirm availability.\n\n"
                "Regards,\nDr. Helena Marsh\nPartner, Marsh & Associates LLP"
            ),
            "received_at": "2024-06-10T11:00:00Z",
            "has_attachment": False,
            "thread_length": 4,
        },
        "ground_truth": {
            "urgency": "urgent",
            "category": "legal",
            "sender_name": "Dr. Helena Marsh",
            "deadline": "today",
            "sentiment": "negative",
            "reply_must_contain": ["DPO", "CTO", "CISO", "confirm", "call"],
            "reply_tone": "urgent_and_compliant",
        },
    },
    {
        "email": {
            "id": "e008",
            "subject": "Cancellation request + refund + formal complaint",
            "sender": "angry.customer@gmail.com",
            "sender_domain": "gmail.com",
            "body": (
                "I have been a customer for three years and this is absolutely unacceptable. "
                "Your service has been down FOUR times this month. I lost important data and "
                "had to manually redo 8 hours of work. I want:\n"
                "1. Immediate cancellation of my subscription\n"
                "2. Full refund for this month ($299)\n"
                "3. Compensation for my lost time\n"
                "4. A written explanation of what went wrong\n\n"
                "If I do not hear back within 24 hours I am filing a complaint with "
                "the consumer protection agency and posting this experience publicly.\n\n"
                "— David Park"
            ),
            "received_at": "2024-06-10T14:55:00Z",
            "has_attachment": False,
            "thread_length": 1,
        },
        "ground_truth": {
            "urgency": "urgent",
            "category": "billing",
            "sender_name": "David Park",
            "deadline": "24 hours",
            "sentiment": "angry",
            "reply_must_contain": ["sorry", "refund", "escalat", "24"],
            "reply_tone": "de-escalation_and_resolution",
        },
    },
    {
        "email": {
            "id": "e009",
            "subject": "Partnership proposal — co-marketing agreement",
            "sender": "partnerships@growthco.com",
            "sender_domain": "growthco.com",
            "body": (
                "Hi there,\n\n"
                "I'm reaching out from GrowthCo, a B2B SaaS platform serving 2,000+ "
                "mid-market companies. We've been following your product closely and believe "
                "there's a strong co-marketing opportunity — our audiences overlap by ~40% "
                "but we don't compete directly.\n\n"
                "We're proposing a joint webinar series and content swap. Our last webinar "
                "drew 1,200 registrants. We'd love to schedule a 30-minute intro call in "
                "the next two weeks.\n\n"
                "Would Wednesday July 3rd or Thursday July 4th work for your team?\n\n"
                "Looking forward to connecting,\nAlex Torres\nHead of Partnerships, GrowthCo"
            ),
            "received_at": "2024-06-09T09:20:00Z",
            "has_attachment": False,
            "thread_length": 1,
        },
        "ground_truth": {
            "urgency": "normal",
            "category": "sales",
            "sender_name": "Alex Torres",
            "deadline": "2024-07-04",
            "sentiment": "positive",
            "reply_must_contain": ["webinar", "call", "July", "interested"],
            "reply_tone": "warm_professional_interest",
        },
    },
]


def get_emails_for_task(task_name: str) -> List[Dict[str, Any]]:
    """Return emails appropriate for the given task difficulty."""
    if task_name == "classify-urgency":
        # Easy: first 3 emails
        return EMAILS[:3]
    elif task_name == "classify-and-extract":
        # Medium: middle 3 emails
        return EMAILS[3:6]
    elif task_name == "full-triage":
        # Hard: last 3 emails
        return EMAILS[6:]
    else:
        return EMAILS