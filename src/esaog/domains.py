"""Declarative domain specifications shared by the ontology builder and the SOST generator.

Each domain extends the same upper ontology (esaog core) with
  * a capability hierarchy (rdfs:subClassOf) plus equivalence axioms,
  * a data-type hierarchy used for typed inputs/outputs,
  * workflow slot templates (task types) with typed I/O and policy defaults.

Nothing here is method-specific: every orchestrator receives the same instances.
"""

# (name, parent or None, label, alt labels, short gloss)
LIT_CAPS = [
    ("LiteratureSearch", None, "literature search", ["bibliographic search"], "retrieve scholarly records"),
    ("BiomedicalSearch", "LiteratureSearch", "biomedical literature search", [], "retrieve biomedical studies"),
    ("PubMedSearch", "BiomedicalSearch", "PubMed search", ["MEDLINE retrieval"], "MeSH-indexed MEDLINE retrieval"),
    ("EmbaseSearch", "BiomedicalSearch", "Embase search", [], "Emtree-indexed Embase retrieval"),
    ("PreprintSearch", "LiteratureSearch", "preprint search", [], "retrieve preprint server records"),
    ("CitationIndexSearch", "LiteratureSearch", "citation index search", [], "forward/backward citation chasing"),
    ("WebSearch", None, "web search", ["internet search"], "general web page search"),
    ("Screening", None, "study screening", [], "include or exclude records"),
    ("TitleAbstractScreening", "Screening", "title and abstract screening", [], "first-pass screening"),
    ("FullTextScreening", "Screening", "full-text screening", [], "second-pass screening"),
    ("EligibilityScreening", "Screening", "eligibility screening", [], "apply eligibility criteria"),
    ("CriteriaBasedScreening", None, "criteria-based record triage", [], "rule-driven inclusion triage"),
    ("EvidenceExtraction", None, "evidence extraction", ["data extraction"], "extract study data"),
    ("PICOExtraction", "EvidenceExtraction", "PICO extraction", [], "population/intervention/comparator/outcome extraction"),
    ("OutcomeDataExtraction", "EvidenceExtraction", "outcome data extraction", [], "effect-size extraction"),
    ("QualityAppraisal", None, "quality appraisal", ["critical appraisal"], "appraise study quality"),
    ("RiskOfBiasAssessment", "QualityAppraisal", "risk of bias assessment", [], "assess risk of bias"),
    ("RoB2Assessment", "RiskOfBiasAssessment", "RoB 2 assessment", [], "randomised-trial risk of bias"),
    ("CochraneRiskOfBias2", None, "Cochrane risk-of-bias tool v2", [], "Cochrane RCT bias domains"),
    ("ROBINSIAssessment", "RiskOfBiasAssessment", "ROBINS-I assessment", [], "non-randomised study bias"),
    ("ReviewQualityAppraisal", "QualityAppraisal", "review quality appraisal", [], "appraise systematic reviews"),
    ("AMSTAR2Appraisal", "ReviewQualityAppraisal", "AMSTAR 2 appraisal", [], "AMSTAR 2 checklist"),
    ("EvidenceSynthesis", None, "evidence synthesis", [], "synthesise findings"),
    ("MetaAnalysis", "EvidenceSynthesis", "meta-analysis", [], "quantitative pooling"),
    ("QuantitativeEvidencePooling", None, "quantitative evidence pooling", [], "statistical pooling of effects"),
    ("NarrativeSynthesis", "EvidenceSynthesis", "narrative synthesis", [], "qualitative narrative summary"),
    ("CitationVerification", None, "citation verification", ["reference checking"], "verify citations"),
    ("FinancialForecasting", None, "financial forecasting", [], "forecast financial indicators"),
]
LIT_EQUIV = [("EligibilityScreening", "CriteriaBasedScreening"), ("RoB2Assessment", "CochraneRiskOfBias2"),
             ("MetaAnalysis", "QuantitativeEvidencePooling")]
LIT_TYPES = [
    ("ResearchQuestion", None), ("BibliographicRecordSet", None), ("PeerReviewedRecordSet", "BibliographicRecordSet"),
    ("PreprintRecordSet", "BibliographicRecordSet"), ("WebPageSet", None), ("ScreenedRecordSet", None),
    ("ExtractionTable", None), ("PICOTable", "ExtractionTable"), ("AppraisalReport", None),
    ("RiskOfBiasReport", "AppraisalReport"), ("ReviewQualityReport", "AppraisalReport"), ("SynthesisReport", None),
    ("MetaAnalysisReport", "SynthesisReport"), ("NarrativeSynthesisReport", "SynthesisReport"),
    ("VerifiedReport", None), ("FinancialForecastReport", None), ("KeywordList", None),
]
LIT_SLOTS = [
    dict(id="search", text="search bibliographic databases for peer reviewed studies on {topic}",
         req=["LiteratureSearch", "BiomedicalSearch", "LiteratureSearch"], inputs=["goal"], in_type="ResearchQuestion",
         out="BibliographicRecordSet", sub_out="PeerReviewedRecordSet", bad_out="WebPageSet", bad_in="KeywordList"),
    dict(id="screen", text="screen retrieved records against the eligibility criteria for {topic}",
         req=["Screening", "EligibilityScreening", "TitleAbstractScreening"], inputs=["search"],
         in_type="BibliographicRecordSet", out="ScreenedRecordSet", sub_out=None, bad_out="KeywordList", bad_in="WebPageSet"),
    dict(id="extract", text="extract study characteristics and outcome data from included studies on {topic}",
         req=["EvidenceExtraction", "PICOExtraction", "EvidenceExtraction"], inputs=["screen"], in_type="ScreenedRecordSet",
         out="ExtractionTable", sub_out="PICOTable", bad_out="KeywordList", bad_in="WebPageSet"),
    dict(id="appraise", text="assess the risk of bias and methodological quality of included studies on {topic}",
         req=["QualityAppraisal", "RiskOfBiasAssessment", "RoB2Assessment"], inputs=["screen"], in_type="ScreenedRecordSet",
         out="AppraisalReport", sub_out="RiskOfBiasReport", bad_out="FinancialForecastReport", bad_in="WebPageSet"),
    dict(id="synth", text="synthesise the extracted evidence and quality appraisal into a pooled summary on {topic}",
         req=["EvidenceSynthesis", "MetaAnalysis", "NarrativeSynthesis"], inputs=["extract", "appraise"],
         in_type="ExtractionTable", in_type2="AppraisalReport", out="SynthesisReport", sub_out="MetaAnalysisReport",
         bad_out="FinancialForecastReport", bad_in="WebPageSet"),
    dict(id="verify", text="verify every citation and reference in the synthesis report on {topic}",
         req=["CitationVerification"], inputs=["synth"], in_type="SynthesisReport", out="VerifiedReport",
         sub_out=None, bad_out="KeywordList", bad_in="WebPageSet"),
]
LIT_TOPICS = ["AI-assisted breast cancer screening", "remote cognitive behavioural therapy for insomnia",
              "statins for primary prevention in older adults", "digital phenotyping for depression relapse",
              "robotic versus laparoscopic colectomy", "school-based physical activity interventions",
              "antibiotic stewardship in primary care", "telemonitoring after heart failure discharge",
              "machine learning sepsis alerts", "mindfulness apps for anxiety"]

SW_CAPS = [
    ("SoftwarePlanning", None, "software planning", [], "plan code changes"),
    ("RequirementsAnalysis", "SoftwarePlanning", "requirements analysis", [], "analyse issue requirements"),
    ("ArchitectureDesign", "SoftwarePlanning", "architecture design", [], "design module structure"),
    ("CodeGeneration", None, "code generation", ["code synthesis"], "write source code"),
    ("PythonCodeGeneration", "CodeGeneration", "Python code generation", [], "write Python code"),
    ("TypedPythonGeneration", "PythonCodeGeneration", "typed Python code generation", [], "mypy-clean Python"),
    ("JavaCodeGeneration", "CodeGeneration", "Java code generation", [], "write Java code"),
    ("CodeSnippetSearch", None, "code snippet search", [], "search snippets on the web"),
    ("SoftwareTesting", None, "software testing", [], "test code changes"),
    ("UnitTesting", "SoftwareTesting", "unit testing", [], "unit-level tests"),
    ("ComponentTesting", None, "component-level verification", [], "isolated component checks"),
    ("PropertyBasedTesting", "SoftwareTesting", "property-based testing", [], "generative property tests"),
    ("MutationTesting", "SoftwareTesting", "mutation testing", [], "mutation-score analysis"),
    ("SecurityReview", None, "security review", [], "review code for vulnerabilities"),
    ("StaticSecurityAnalysis", "SecurityReview", "static security analysis", ["SAST"], "static vulnerability scanning"),
    ("SASTScan", None, "SAST scan", [], "static application security testing"),
    ("SecretScanning", "SecurityReview", "secret scanning", [], "detect leaked credentials"),
    ("PenetrationTesting", "SecurityReview", "penetration testing", [], "dynamic attack simulation"),
    ("DependencyAnalysis", None, "dependency analysis", [], "analyse third-party dependencies"),
    ("SBOMGeneration", "DependencyAnalysis", "SBOM generation", [], "software bill of materials"),
    ("DependencyVulnerabilityAudit", "DependencyAnalysis", "dependency vulnerability audit", [], "CVE audit of packages"),
    ("LicenseAudit", "DependencyAnalysis", "license audit", [], "licence compliance of packages"),
    ("RepositoryOperation", None, "repository operation", [], "operate on the repository"),
    ("PullRequestCreation", "RepositoryOperation", "pull request creation", [], "open a pull request"),
    ("BranchManagement", "RepositoryOperation", "branch management", [], "manage branches"),
    ("MarketingCopywriting", None, "marketing copywriting", [], "write product marketing copy"),
]
SW_EQUIV = [("UnitTesting", "ComponentTesting"), ("StaticSecurityAnalysis", "SASTScan")]
SW_TYPES = [
    ("IssueReport", None), ("DesignPlan", None), ("SourcePatch", None), ("TypedSourcePatch", "SourcePatch"),
    ("CodeSnippetText", None), ("TestReport", None), ("CoverageTestReport", "TestReport"), ("SecurityReport", None),
    ("SASTReport", "SecurityReport"), ("DependencyReport", None), ("SBOMDocument", "DependencyReport"),
    ("PullRequest", None), ("MarketingText", None), ("ChatTranscript", None),
]
SW_SLOTS = [
    dict(id="plan", text="analyse the issue report and plan the code change for {topic}", req=["SoftwarePlanning", "RequirementsAnalysis"],
         inputs=["goal"], in_type="IssueReport", out="DesignPlan", sub_out=None, bad_out="MarketingText", bad_in="ChatTranscript"),
    dict(id="code", text="implement the planned source code patch for {topic}", req=["CodeGeneration", "PythonCodeGeneration", "CodeGeneration"],
         inputs=["plan"], in_type="DesignPlan", out="SourcePatch", sub_out="TypedSourcePatch", bad_out="CodeSnippetText", bad_in="ChatTranscript"),
    dict(id="test", text="run unit tests against the source patch for {topic}", req=["SoftwareTesting", "UnitTesting", "SoftwareTesting"],
         inputs=["code"], in_type="SourcePatch", out="TestReport", sub_out="CoverageTestReport", bad_out="MarketingText", bad_in="CodeSnippetText"),
    dict(id="security", text="perform a static security review of the source patch for {topic}", req=["SecurityReview", "StaticSecurityAnalysis"],
         inputs=["code"], in_type="SourcePatch", out="SecurityReport", sub_out="SASTReport", bad_out="MarketingText", bad_in="CodeSnippetText"),
    dict(id="deps", text="analyse third party dependencies introduced by the patch for {topic}", req=["DependencyAnalysis", "DependencyVulnerabilityAudit"],
         inputs=["code"], in_type="SourcePatch", out="DependencyReport", sub_out="SBOMDocument", bad_out="MarketingText", bad_in="ChatTranscript"),
    dict(id="pr", text="open a pull request with the patch, test and security reports for {topic}", req=["RepositoryOperation", "PullRequestCreation"],
         inputs=["test", "security"], in_type="TestReport", in_type2="SecurityReport", out="PullRequest", sub_out=None,
         bad_out="MarketingText", bad_in="ChatTranscript"),
]
SW_TOPICS = ["rate limiting in the payments API", "OAuth token refresh bug", "CSV export memory leak",
             "timezone handling in the scheduler", "search pagination regression", "PDF invoice rendering",
             "feature flag cleanup", "retry logic for the webhook client", "database migration for audit logs",
             "input sanitisation in the upload endpoint"]

ENT_CAPS = [
    ("DocumentExtraction", None, "document extraction", [], "extract fields from documents"),
    ("InvoiceExtraction", "DocumentExtraction", "invoice extraction", [], "extract invoice fields"),
    ("OCRInvoiceExtraction", "InvoiceExtraction", "OCR invoice extraction", [], "OCR-based invoice capture"),
    ("BillDataCapture", None, "bill data capture", [], "capture billing data"),
    ("ContractExtraction", "DocumentExtraction", "contract extraction", [], "extract contract clauses"),
    ("DataValidation", None, "data validation", [], "validate extracted records"),
    ("SchemaValidation", "DataValidation", "schema validation", [], "validate against schema"),
    ("ThreeWayMatch", "DataValidation", "three-way match", [], "PO/receipt/invoice matching"),
    ("ComplianceCheck", None, "compliance check", [], "check regulatory compliance"),
    ("AMLScreening", "ComplianceCheck", "AML screening", [], "anti-money-laundering screening"),
    ("AntiMoneyLaunderingCheck", None, "anti-money-laundering check", [], "AML rule evaluation"),
    ("SanctionsScreening", "ComplianceCheck", "sanctions screening", [], "sanctions list screening"),
    ("GDPRCheck", "ComplianceCheck", "GDPR check", [], "data-protection compliance"),
    ("Approval", None, "approval", [], "approve or reject"),
    ("ManagerApproval", "Approval", "manager approval", [], "line-manager sign-off"),
    ("FinanceApproval", "Approval", "finance approval", [], "finance controller sign-off"),
    ("Notification", None, "notification", [], "notify stakeholders"),
    ("EmailNotification", "Notification", "email notification", [], "send e-mail"),
    ("ChatNotification", "Notification", "chat notification", [], "post chat message"),
    ("RecordsArchiving", None, "records archiving", [], "archive records"),
    ("SocialMediaPosting", None, "social media posting", [], "post on social media"),
]
ENT_EQUIV = [("AMLScreening", "AntiMoneyLaunderingCheck"), ("InvoiceExtraction", "BillDataCapture")]
ENT_TYPES = [
    ("ScannedDocument", None), ("ExtractedRecord", None), ("InvoiceRecord", "ExtractedRecord"), ("ValidatedRecord", None),
    ("ComplianceReport", None), ("AMLReport", "ComplianceReport"), ("ApprovalDecision", None),
    ("NotificationReceipt", None), ("ArchiveReceipt", None), ("PlainTextSummary", None), ("SocialPost", None),
]
ENT_SLOTS = [
    dict(id="extract", text="extract the fields of the incoming invoice document for {topic}", req=["DocumentExtraction", "InvoiceExtraction"],
         inputs=["goal"], in_type="ScannedDocument", out="ExtractedRecord", sub_out="InvoiceRecord", bad_out="PlainTextSummary", bad_in="SocialPost"),
    dict(id="validate", text="validate the extracted record against purchase order data for {topic}", req=["DataValidation", "ThreeWayMatch", "DataValidation"],
         inputs=["extract"], in_type="ExtractedRecord", out="ValidatedRecord", sub_out=None, bad_out="PlainTextSummary", bad_in="SocialPost"),
    dict(id="comply", text="run regulatory compliance screening on the validated record for {topic}", req=["ComplianceCheck", "AMLScreening", "ComplianceCheck"],
         inputs=["validate"], in_type="ValidatedRecord", out="ComplianceReport", sub_out="AMLReport", bad_out="PlainTextSummary", bad_in="SocialPost"),
    dict(id="approve", text="obtain approval for the payment based on the compliance report for {topic}", req=["Approval", "ManagerApproval", "FinanceApproval"],
         inputs=["comply"], in_type="ComplianceReport", out="ApprovalDecision", sub_out=None, bad_out="PlainTextSummary", bad_in="SocialPost"),
    dict(id="notify", text="notify the requester about the approval decision for {topic}", req=["Notification", "EmailNotification"],
         inputs=["approve"], in_type="ApprovalDecision", out="NotificationReceipt", sub_out=None, bad_out="SocialPost", bad_in="PlainTextSummary"),
    dict(id="archive", text="archive the approved record and decision for {topic}", req=["RecordsArchiving"],
         inputs=["approve"], in_type="ApprovalDecision", out="ArchiveReceipt", sub_out=None, bad_out="PlainTextSummary", bad_in="SocialPost"),
]
ENT_TOPICS = ["supplier invoice INV-2291 from a logistics vendor", "cross-border consulting payment", "cloud hosting renewal invoice",
              "equipment lease payment", "marketing agency retainer", "customer refund above threshold",
              "new vendor onboarding payment", "travel reimbursement batch", "software licence true-up", "insurance premium payment"]

DOMAINS = {
    "literature": dict(prefix="lit", ns="https://w3id.org/esaog/lit#", caps=LIT_CAPS, equiv=LIT_EQUIV, types=LIT_TYPES,
                       slots=LIT_SLOTS, topics=LIT_TOPICS, distractor_cap="FinancialForecasting", generic_cap="WebSearch",
                       goal_verb="Produce a verified evidence synthesis on", pii_slots=[]),
    "software": dict(prefix="sw", ns="https://w3id.org/esaog/sw#", caps=SW_CAPS, equiv=SW_EQUIV, types=SW_TYPES,
                     slots=SW_SLOTS, topics=SW_TOPICS, distractor_cap="MarketingCopywriting", generic_cap="CodeSnippetSearch",
                     goal_verb="Deliver a reviewed pull request fixing", pii_slots=["security"]),
    "enterprise": dict(prefix="ent", ns="https://w3id.org/esaog/ent#", caps=ENT_CAPS, equiv=ENT_EQUIV, types=ENT_TYPES,
                       slots=ENT_SLOTS, topics=ENT_TOPICS, distractor_cap="SocialMediaPosting", generic_cap="SocialMediaPosting",
                       goal_verb="Process and approve the", pii_slots=["extract", "comply"]),
}

# Capabilities that, by obligation policy, must be realised by a HumanActor.
HUMAN_ONLY_CAPS = {"Approval"}


def cap_index(domain):
    d = DOMAINS[domain]
    parent = {c[0]: c[1] for c in d["caps"]}
    label = {c[0]: c[2] for c in d["caps"]}
    alt = {c[0]: c[3] for c in d["caps"]}
    gloss = {c[0]: c[4] for c in d["caps"]}
    children = {}
    for c, p in parent.items():
        children.setdefault(p, []).append(c)
    return parent, children, label, alt, gloss


def type_index(domain):
    return {t[0]: t[1] for t in DOMAINS[domain]["types"]}
