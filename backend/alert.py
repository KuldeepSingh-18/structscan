"""
Alert & Severity System
- Severity scoring with 4 levels
- Crack-type-specific repair recommendations
- Civil engineering helplines (India)
- Sound alert flag
"""

SEVERITY_TABLE = [
    (0, 20, "SAFE", "#16a34a",
     "Structure appears stable. No immediate damage detected.",
     [
         "Schedule routine visual inspection every 6 months",
         "Document current condition with dated photographs",
         "Ensure proper drainage around the structure",
         "Monitor any hairline cracks for progression",
         "Maintain inspection records for compliance",
     ]),
    (20, 30, "RISK", "#65a30d",
     "Minor surface indicators detected. Monitor regularly.",
     [
         "Conduct close-up inspection of flagged zones within 2 weeks",
         "Measure and record crack widths for comparison",
         "Apply surface sealant to prevent moisture ingress",
         "Improve drainage near affected areas",
         "Re-assess in 30 days to check for progression",
     ]),
    (30, 50, "HIGH RISK", "#d97706",
     "Significant indicators detected. Professional inspection advised.",
     [
         "Engage a licensed structural engineer within 7 days",
         "Fill visible cracks with polyurethane or epoxy injection",
         "Restrict heavy loading near damaged zones",
         "Install crack monitors (tell-tales) for ongoing tracking",
         "Document all findings with measurements and photos",
         "Check for water ingress and rebar corrosion nearby",
     ]),
    (50, 75, "DANGEROUS", "#ea580c",
     "Serious structural damage. Integrity may be compromised.",
     [
         "Immediate structural engineering assessment required",
         "Restrict access or loading in affected area NOW",
         "Apply temporary shoring or propping if safe to do so",
         "Begin formal repair planning with certified engineer",
         "Notify building inspector and safety officer",
         "Check for secondary damage: spalling, delamination, rebar exposure",
     ]),
    (75, 101, "CRITICAL", "#dc2626",
     "SEVERE structural damage. Emergency action required immediately.",
     [
         "⚠️ EVACUATE the area immediately if safety risk exists",
         "Contact a structural engineer URGENTLY — do not delay",
         "Do NOT use or load the affected structure",
         "Emergency shoring or propping is likely required",
         "Notify building inspector and local authorities",
         "Arrange emergency repair with certified contractors",
         "Begin insurance and damage documentation immediately",
     ]),
]

CRACK_SOLUTIONS = {
    "Hairline Crack": [
        "Apply flexible polyurethane sealant or epoxy-based crack filler",
        "Monitor crack width every 2 weeks using a crack gauge or ruler",
        "Ensure no water ingress through the crack — apply waterproof coating",
        "Photograph and document crack dimensions for progression tracking",
        "Check for thermal expansion — ensure adequate expansion joints nearby",
        "Clean crack surface before sealing to ensure proper adhesion",
    ],
    "Linear Structural Crack": [
        "Epoxy injection grouting is the recommended repair method",
        "Install crack monitors (tell-tales) to measure any progression",
        "Licensed structural engineer must assess load-bearing capacity",
        "Identify root cause: foundation settlement, thermal movement, or overload",
        "Temporarily restrict heavy loads near the cracked zone",
        "Check adjacent areas for secondary cracking or deformation",
        "Prepare repair documentation for insurance and compliance records",
    ],
    "Diagonal Shear Crack": [
        "URGENT: Diagonal cracks indicate possible shear stress failure",
        "Reduce structural load immediately if safe to do so",
        "Structural engineer assessment must happen within 24 hours",
        "Carbon fiber reinforcement strips (CFRP) may be required for repair",
        "Check foundation and soil stability for subsidence signs",
        "Install temporary shoring or propping if crack is widening",
        "Notify building inspector and record in maintenance log",
    ],
    "Spalling / Delamination": [
        "Remove all loose and delaminated concrete material with chisel",
        "Apply bonding primer/agent before patching",
        "Patch with polymer-modified cementitious repair mortar",
        "Inspect underlying rebar for corrosion — treat with anti-corrosion epoxy coating",
        "Re-surface with protective cementitious or elastomeric coating after repair",
        "Identify cause of spalling: freeze-thaw cycles, rebar corrosion, or alkali-silica reaction",
        "Consider applying penetrating sealant to prevent moisture ingress",
    ],
    "Vertical / Horizontal Crack": [
        "Identify root cause: foundation settlement, thermal movement, or overloading",
        "Use crack stitching technique with stainless steel stitching bars for structural cracks",
        "Apply flexible waterproof coating over repaired surface",
        "Monitor crack for continued movement over 4-6 weeks",
        "Check for differential settlement in foundation — may need underpinning",
        "Install tell-tales or crack monitors for objective measurement",
        "Consult a structural engineer if crack width exceeds 0.3mm",
    ],
}

HELPLINES = [
    {"name": "National Disaster Management Authority (NDMA)", "number": "1078", "note": "24x7 emergency"},
    {"name": "IIT Structural Engineering Helpdesk", "number": "+91-11-2659-7697", "note": "Expert consultation"},
    {"name": "Bureau of Indian Standards (BIS)", "number": "+91-11-2323-7991", "note": "Building code queries"},
    {"name": "State PWD (Public Works Dept)", "number": "1800-180-5400", "note": "Government infrastructure"},
    {"name": "National Emergency", "number": "112", "note": "Police/Fire/Ambulance"},
]


class AlertSystem:
    def generate_alert(self, result: dict) -> dict:
        score = result.get("severity_score", 0)
        label = result.get("label", "Non-Cracked")
        crack_info = result.get("crack_info", {})
        crack_type = crack_info.get("crack_type", "None")

        severity_level = "SAFE"
        severity_color = "#22c55e"
        alert_message = ""
        base_solutions = []

        for low, high, level, color, message, solutions in SEVERITY_TABLE:
            if low <= score < high:
                severity_level = level
                severity_color = color
                alert_message = message
                base_solutions = solutions[:]
                break

        # Add crack-type specific solutions
        crack_solutions = CRACK_SOLUTIONS.get(crack_type, [])
        all_solutions = base_solutions + crack_solutions

        # Show helplines only for high severity
        show_helplines = score >= 50
        helplines = HELPLINES if show_helplines else []

        return {
            "severity_level": severity_level,
            "severity_color": severity_color,
            "alert_message": alert_message,
            "solutions": all_solutions,
            "helplines": helplines,
            "play_alert_sound": label == "Cracked" and score >= 30,
            "damage_detected": label == "Cracked",
        }
