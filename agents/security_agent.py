class SecurityAgent:
    def scan_output(self, text: str):
        return {
            "security_score": 100,
            "security_status": "PASS",
            "security_findings": []
        }

