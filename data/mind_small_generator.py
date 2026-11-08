"""
MIND Small Dataset Generator
===============================
Generates a realistic MIND-small-format dataset with:
- 1,000+ news articles across 10 categories and 30+ subcategories
- 500 users with diverse reading personas
- Realistic click patterns, user histories, and impression logs
- Entity Wikidata IDs for HIN knowledge graph connections

Format matches Microsoft MIND dataset exactly:
  news.tsv    -> news_id, category, subcategory, title, abstract, url, title_entities, abstract_entities
  behaviors.tsv -> impression_id, user_id, time, history, impressions
"""

import os
import json
import random
import hashlib
from datetime import datetime, timedelta

random.seed(42)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ───────────────────────────────────────────────────────────────────────────────
# Knowledge Graph: Wikidata Entities
# ───────────────────────────────────────────────────────────────────────────────
ENTITIES = {
    # Technology
    "Q11660":  "Artificial Intelligence",
    "Q65640329": "OpenAI",
    "Q312":    "Apple Inc.",
    "Q11032":  "Newspaper",
    "Q8078":   "Cybersecurity",
    "Q483247": "Cloud Computing",
    "Q11012":  "Robotics",
    "Q177777": "Logistics",
    "Q7367":   "Software",
    "Q9143":   "Programming Language",
    "Q5300":   "Silicon",
    "Q80831":  "Microsoft",
    "Q95":     "Google",
    "Q37156":  "Meta Platforms",
    "Q46812":  "Amazon",
    "Q21146257": "Machine Learning",
    "Q131723": "Bitcoin",
    "Q11190":  "Medicine",
    "Q186447": "James Webb Space Telescope",
    "Q318":    "Galaxy",
    "Q11438":  "Plastic",
    "Q8047":   "Enzyme",
    "Q413":    "Physics",
    "Q131765": "mRNA",
    "Q12111":  "Cancer",
    "Q131758": "Nutrition",
    "Q21114":  "Health",
    "Q7889":   "Video Game",
    "Q1889":   "Gaming",
    "Q571":    "Book",
    "Q24925":  "Science Fiction",
    "Q11424":  "Film",
    "Q638":    "Music",
    "Q41254":  "Grammy Award",
    "Q43203":  "Christopher Nolan",
    # Sports
    "Q8682":   "Real Madrid",
    "Q2718":   "Football",
    "Q121776": "Los Angeles Lakers",
    "Q5372":   "Basketball",
    "Q9842":   "Cricket",
    "Q668":    "India",
    "Q171321": "Formula One",
    "Q169898": "Ferrari",
    "Q850239": "Tennis",
    "Q62024195": "Carlos Alcaraz",
    "Q23808478": "LeBron James",
    "Q1348952": "Novak Djokovic",
    "Q7336":   "Golf",
    "Q131436": "Swimming",
    "Q5500":   "Athletics",
    # Finance
    "Q53536":  "Federal Reserve",
    "Q8142":   "Economy",
    "Q13479982": "Cryptocurrency",
    "Q11442":  "Semiconductor",
    "Q11691":  "Stock Market",
    "Q68947":  "Real Estate",
    "Q4830453": "Business",
    "Q217409": "Venture Capital",
    "Q12705":  "Clean Energy",
    "Q11430":  "Gold",
    "Q6166":   "Bond",
    "Q2516":   "Inflation",
    "Q207477": "Hedge Fund",
    "Q5327":   "Currency",
    # Science
    "Q41710":  "Biology",
    "Q7187":   "Gene",
    "Q172807": "CRISPR",
    "Q10366":  "Vaccine",
    "Q11379":  "Atom",
    "Q11406":  "Thermodynamics",
    "Q7777":   "Astronomy",
    "Q11348":  "Chemistry",
    "Q1343":   "Mars",
    "Q596":    "Moon",
    "Q525":    "Sun",
    "Q2566":   "Climate",
    "Q11288":  "Ecology",
    # Politics
    "Q7279":   "Political Party",
    "Q6581097": "Democracy",
    "Q7278":   "Politics",
    "Q161174": "United Nations",
    "Q1065":   "NATO",
    "Q828":    "Europe",
    "Q30":     "United States",
    "Q148":    "China",
    "Q159":    "Russia",
    "Q17":     "Japan",
    # Lifestyle
    "Q174":    "Food",
    "Q11032":  "Travel",
    "Q5119":   "Fashion",
    "Q13406463": "Recipe",
    "Q11190":  "Fitness",
    "Q349":    "Sport",
}

# ───────────────────────────────────────────────────────────────────────────────
# Article templates per category
# ───────────────────────────────────────────────────────────────────────────────
ARTICLES = []

def make_entity(wid):
    return {"WikidataId": wid, "Label": ENTITIES.get(wid, wid)}

# ── TECHNOLOGY (200 articles) ──────────────────────────────────────────────────
TECH_TEMPLATES = [
    # AI
    ("technology", "artificial-intelligence",
     "OpenAI Releases GPT-5 with Unprecedented Reasoning Abilities",
     "The latest language model demonstrates advanced multi-step reasoning across mathematics, code, and science.",
     ["Q65640329", "Q11660"], ["Q21146257", "Q7367"]),
    ("technology", "artificial-intelligence",
     "Google DeepMind Achieves Breakthrough in Protein Structure Prediction",
     "New AlphaFold variant predicts intrinsically disordered protein regions with high accuracy.",
     ["Q95", "Q11660"], ["Q41710", "Q7187"]),
    ("technology", "artificial-intelligence",
     "Meta Releases Open Source Large Language Model Beating Proprietary Rivals",
     "Researchers benchmark the new model against GPT-4 and Claude on standardized evaluation suites.",
     ["Q37156", "Q11660"], ["Q21146257", "Q9143"]),
    ("technology", "artificial-intelligence",
     "Graph Neural Networks Revolutionize Drug Discovery Pipeline",
     "Biotech researchers use geometric deep learning to simulate molecular bond interactions.",
     ["Q11660", "Q11190"], ["Q21146257", "Q41710"]),
    ("technology", "artificial-intelligence",
     "AI System Masters Complex Strategy Game Without Human Data",
     "Reinforcement learning agent achieves superhuman performance through pure self-play.",
     ["Q11660", "Q21146257"], ["Q7367", "Q95"]),
    ("technology", "artificial-intelligence",
     "Microsoft Integrates Generative AI Into Office Productivity Suite",
     "Copilot features automate document summarization, email drafting, and spreadsheet analysis.",
     ["Q80831", "Q11660"], ["Q21146257", "Q7367"]),
    ("technology", "artificial-intelligence",
     "Neural Architecture Search Automates Deep Learning Model Design",
     "AutoML techniques reduce expert engineering time for custom neural network architectures.",
     ["Q11660", "Q21146257"], ["Q9143", "Q7367"]),
    ("technology", "artificial-intelligence",
     "Multimodal AI Model Understands Images, Text, and Audio Simultaneously",
     "Foundation models trained on diverse data modalities show emergent cross-modal reasoning.",
     ["Q11660", "Q95"], ["Q21146257", "Q7367"]),
    # Cloud / Infra
    ("technology", "cloud-computing",
     "AWS Launches Next-Generation Graviton Processor for Cloud Workloads",
     "The new ARM-based chip offers 40% better price-performance for containerized microservices.",
     ["Q46812", "Q483247"], ["Q5300", "Q7367"]),
    ("technology", "cloud-computing",
     "Global Cyberattack Targets Major Cloud Infrastructure Services",
     "Security researchers identify zero-day vulnerabilities in containerized microservices across leading providers.",
     ["Q8078", "Q483247"], ["Q7367", "Q80831"]),
    ("technology", "cloud-computing",
     "Kubernetes Adoption Surges as Enterprises Embrace Multi-Cloud Strategy",
     "Container orchestration platform becomes standard for deploying scalable microservices workloads.",
     ["Q483247", "Q7367"], ["Q46812", "Q95"]),
    # Gadgets
    ("technology", "gadgets",
     "Apple Unveils M4 Pro Silicon Chip for High-Performance Workstations",
     "Apple has launched its latest silicon architecture offering unprecedented hardware acceleration for AI.",
     ["Q312", "Q5300"], ["Q11660", "Q7367"]),
    ("technology", "gadgets",
     "Samsung Galaxy S25 Ultra Sets New Benchmark for Mobile Photography",
     "Advanced computational photography pipeline with AI enhancement produces DSLR-quality images.",
     ["Q11660", "Q7367"], ["Q5300", "Q312"]),
    ("technology", "gadgets",
     "Meta Quest 4 Launches with Full Body Tracking and Neural Interface",
     "Augmented and virtual reality converge in next-generation mixed reality headset.",
     ["Q37156", "Q11660"], ["Q7889", "Q7367"]),
    # Robotics
    ("technology", "robotics",
     "Humanoid Robots Enter Automated Logistics Warehouses Globally",
     "Autonomous bipedal robots assist human workers in packing and sorting shipping parcels.",
     ["Q11012", "Q177777"], ["Q11660", "Q21146257"]),
    ("technology", "robotics",
     "Boston Dynamics Atlas Robot Demonstrates Advanced Dexterous Manipulation",
     "New algorithms allow precise handling of irregularly shaped objects in unstructured environments.",
     ["Q11012", "Q11660"], ["Q21146257", "Q177777"]),
    # Cybersecurity
    ("technology", "cybersecurity",
     "Ransomware Group Targets Critical Infrastructure in Coordinated Attack",
     "Energy and water utilities across multiple countries report systems encrypted by sophisticated malware.",
     ["Q8078", "Q483247"], ["Q7367", "Q95"]),
    ("technology", "cybersecurity",
     "Zero Trust Architecture Becomes Mandatory for Federal Agencies",
     "Government directive requires all departments to implement identity-based access controls.",
     ["Q8078", "Q7367"], ["Q483247", "Q30"]),
    ("technology", "cybersecurity",
     "Quantum Cryptography Promises Unbreakable Encryption for Banking",
     "Financial institutions pilot quantum key distribution networks for transaction security.",
     ["Q8078", "Q11660"], ["Q11379", "Q4830453"]),
    # Blockchain
    ("technology", "blockchain",
     "Ethereum Layer 2 Network Processes Record Daily Transactions",
     "ZK-rollup technology dramatically reduces gas fees while maintaining decentralized security.",
     ["Q131723", "Q13479982"], ["Q7367", "Q4830453"]),
    ("technology", "blockchain",
     "Central Bank Digital Currencies Pilot Launches in Major Economies",
     "Digital yuan, euro, and dollar trials test programmable money for retail and wholesale markets.",
     ["Q5327", "Q13479982"], ["Q53536", "Q4830453"]),
]

SPORTS_TEMPLATES = [
    ("sports", "football",
     "Real Madrid Secures Dramatic Champions League Victory in Extra Time",
     "A stoppage time strike seals a stunning comeback in European football quarterfinals.",
     ["Q8682", "Q2718"], ["Q171321", "Q349"]),
    ("sports", "football",
     "Premier League Title Race Enters Final Week with Three Teams Level",
     "Goal difference may decide championship as top clubs face decisive fixtures.",
     ["Q2718", "Q349"], ["Q8682", "Q5500"]),
    ("sports", "football",
     "World Cup Qualifying Draw Sets Stage for Historic Continental Battles",
     "Dramatic pots create potential super-group featuring three former champions.",
     ["Q2718", "Q668"], ["Q349", "Q30"]),
    ("sports", "football",
     "Transfer Window Records Broken as Clubs Splash Billion in Summer Spending",
     "Elite clubs break previous records pursuing top attacking talent from domestic leagues.",
     ["Q2718", "Q8682"], ["Q4830453", "Q349"]),
    ("sports", "basketball",
     "Lakers Stage Remarkable Fourth Quarter Comeback Against Golden State",
     "LeBron James and Anthony Davis combine for 65 points in Western Conference showdown.",
     ["Q121776", "Q5372"], ["Q23808478", "Q349"]),
    ("sports", "basketball",
     "NBA Draft Produces Strongest Rookie Class in Decade According to Scouts",
     "International players dominate top picks as global basketball talent development matures.",
     ["Q5372", "Q349"], ["Q121776", "Q21146257"]),
    ("sports", "cricket",
     "India Clinches Thrilling T20 World Cup Final Against South Africa",
     "Sensational bowling in the death overs leads India to historic championship triumph.",
     ["Q9842", "Q668"], ["Q349", "Q5500"]),
    ("sports", "cricket",
     "Test Cricket Renaissance as Young Nations Challenge Established Powers",
     "Afghanistan and Ireland produce stunning upsets against traditional cricket strongholds.",
     ["Q9842", "Q349"], ["Q668", "Q2566"]),
    ("sports", "formula1",
     "Ferrari Dominates Italian Grand Prix With Masterful Strategic Call",
     "Charles Leclerc holds off McLaren drivers to claim emotional home victory at Monza.",
     ["Q171321", "Q169898"], ["Q349", "Q5500"]),
    ("sports", "formula1",
     "Red Bull Racing Locks Out Front Row in Qualifying for Monaco Grand Prix",
     "Verstappen edges Hamilton by just twelve thousandths in a breathtaking session.",
     ["Q171321", "Q349"], ["Q169898", "Q5500"]),
    ("sports", "tennis",
     "Carlos Alcaraz Defends Grand Slam Title in Five Set Marathon Final",
     "Spanish prodigy outlasts opponent in grueling five-hour final on court central.",
     ["Q850239", "Q62024195"], ["Q349", "Q5500"]),
    ("sports", "tennis",
     "Djokovic Chases Calendar Slam at US Open After French Open Victory",
     "Serbian champion targets unprecedented achievement of winning all four majors in single season.",
     ["Q850239", "Q1348952"], ["Q349", "Q62024195"]),
    ("sports", "golf",
     "Tiger Woods Makes Emotional Return at Masters Tournament Augusta",
     "Golf legend fights back from injury to contend at iconic major championship venue.",
     ["Q7336", "Q349"], ["Q5500", "Q23808478"]),
    ("sports", "swimming",
     "World Record Falls in 100m Freestyle at Olympic Trials",
     "New champion shatters mark that stood for twelve years with stunning race execution.",
     ["Q131436", "Q349"], ["Q5500", "Q668"]),
    ("sports", "athletics",
     "Marathon World Record Broken at Berlin with Sub-Two Hour Pace",
     "Kenyan runner achieves historic milestone in conditions optimized for peak performance.",
     ["Q5500", "Q349"], ["Q668", "Q131436"]),
]

FINANCE_TEMPLATES = [
    ("finance", "markets",
     "Federal Reserve Holds Interest Rates Steady Amid Economic Resilience",
     "Central bank chair indicates monetary policy is well positioned to control inflation.",
     ["Q53536", "Q8142"], ["Q2516", "Q4830453"]),
    ("finance", "markets",
     "Wall Street Hits All-Time High as Tech Earnings Beat Expectations",
     "S&P 500 and Nasdaq surge after major technology companies report stronger than forecast results.",
     ["Q11691", "Q8142"], ["Q4830453", "Q11442"]),
    ("finance", "markets",
     "Bond Market Signals Recession Concerns as Yield Curve Inverts",
     "Investors flee to safe haven assets amid growing fears of economic slowdown.",
     ["Q6166", "Q8142"], ["Q53536", "Q2516"]),
    ("finance", "crypto",
     "Bitcoin Surges Past Milestone High as Institutional Inflows Accelerate",
     "Exchange-traded fund activity and digital asset regulatory clarity drive market momentum.",
     ["Q131723", "Q13479982"], ["Q11691", "Q4830453"]),
    ("finance", "crypto",
     "Ethereum Merge Upgrade Cuts Energy Consumption by Ninety Percent",
     "Proof of stake transition makes second largest blockchain dramatically more sustainable.",
     ["Q13479982", "Q131723"], ["Q2566", "Q7367"]),
    ("finance", "crypto",
     "SEC Approves Spot Crypto ETF Opening Market to Retail Investors",
     "Landmark regulatory approval allows ordinary investors exposure through traditional brokerage accounts.",
     ["Q13479982", "Q11691"], ["Q131723", "Q4830453"]),
    ("finance", "stocks",
     "Semiconductor Stocks Rally Following Strong Global Cloud Demand Reports",
     "Chipmakers surge after earnings demonstrate robust corporate capital expenditure on servers.",
     ["Q11442", "Q11691"], ["Q483247", "Q4830453"]),
    ("finance", "stocks",
     "Energy Stocks Surge as OPEC+ Extends Production Cuts Into Next Year",
     "Oil price rally benefits integrated majors while clean energy transition debate continues.",
     ["Q11691", "Q4830453"], ["Q12705", "Q2566"]),
    ("finance", "real-estate",
     "Commercial Real Estate Adapts to Hybrid Work Trends Post-Pandemic",
     "Office building landlords renovate properties into multi-use spaces with wellness amenities.",
     ["Q68947", "Q4830453"], ["Q2516", "Q8142"]),
    ("finance", "real-estate",
     "Housing Market Cools as Mortgage Rates Reach Decade High",
     "Affordability crisis forces first-time buyers from market as inventory remains critically low.",
     ["Q68947", "Q2516"], ["Q8142", "Q53536"]),
    ("finance", "startup",
     "Venture Capital Focuses Investment on Sustainable Energy and Climate Tech",
     "Clean energy startups draw record early-stage investment into sodium battery technology.",
     ["Q217409", "Q12705"], ["Q4830453", "Q2566"]),
    ("finance", "startup",
     "Fintech Unicorns Face Valuation Resets in Challenging Capital Environment",
     "Late stage companies accept down rounds as investor emphasis shifts to profitability.",
     ["Q217409", "Q4830453"], ["Q11691", "Q13479982"]),
    ("finance", "gold",
     "Gold Price Hits Record as Central Banks Diversify Away from Dollar",
     "Emerging market central banks accelerate gold purchases amid de-dollarization trend.",
     ["Q11430", "Q5327"], ["Q53536", "Q8142"]),
]

HEALTH_TEMPLATES = [
    ("health", "medicine",
     "New mRNA Vaccine Shows Promise in Targeted Cancer Immunotherapy Trials",
     "Clinical trials report durable immune responses against difficult-to-treat solid tumor malignancies.",
     ["Q12111", "Q131765"], ["Q10366", "Q41710"]),
    ("health", "medicine",
     "CRISPR Gene Editing Cures Inherited Blood Disorders in Clinical Trial",
     "Patients with sickle cell disease and thalassemia achieve functional cures after one treatment.",
     ["Q172807", "Q7187"], ["Q12111", "Q41710"]),
    ("health", "medicine",
     "Alzheimer Drug Shows Cognitive Decline Reduction in Phase Three Trial",
     "Antibody therapy targeting amyloid plaques demonstrates statistically significant slowing of disease.",
     ["Q11190", "Q12111"], ["Q131765", "Q41710"]),
    ("health", "wellness",
     "Intermittent Fasting and Circadian Rhythms: What the Science Shows",
     "Nutritional scientists study how meal timing impacts metabolic markers and cellular repair.",
     ["Q131758", "Q21114"], ["Q11190", "Q41710"]),
    ("health", "wellness",
     "Exercise Prescription Outperforms Antidepressants for Mild Depression",
     "Landmark meta-analysis finds structured aerobic exercise produces equivalent mental health outcomes.",
     ["Q349", "Q21114"], ["Q131758", "Q11190"]),
    ("health", "wellness",
     "Sleep Science Reveals Critical Window for Brain Waste Clearance",
     "Glymphatic system most active during deep slow wave sleep stages according to neuroimaging study.",
     ["Q21114", "Q131758"], ["Q11190", "Q41710"]),
    ("health", "nutrition",
     "Plant-Based Diets Linked to Lower Cardiovascular Disease Risk",
     "Prospective cohort study tracks 100,000 participants over twenty years revealing dietary patterns.",
     ["Q131758", "Q21114"], ["Q11190", "Q174"]),
    ("health", "nutrition",
     "Ultra-Processed Food Consumption Accelerates Biological Aging Markers",
     "Epigenetic clock analysis reveals diet quality strongly predicts cellular aging trajectories.",
     ["Q131758", "Q11190"], ["Q174", "Q21114"]),
    ("health", "mental-health",
     "Psilocybin Therapy Receives FDA Breakthrough Designation for PTSD",
     "Phase two results show rapid and durable symptom reduction after just two guided sessions.",
     ["Q11190", "Q12111"], ["Q131765", "Q21114"]),
    ("health", "mental-health",
     "Youth Mental Health Crisis Drives Record Demand for Online Therapy",
     "Telehealth platforms report triple-digit growth as adolescent anxiety and depression rates climb.",
     ["Q21114", "Q11190"], ["Q7367", "Q21146257"]),
]

SCIENCE_TEMPLATES = [
    ("science", "space",
     "James Webb Space Telescope Discovers Ancient Galaxy Near Cosmic Dawn",
     "Astronomers observe stellar formations dating back to just hundreds of millions years after Big Bang.",
     ["Q186447", "Q318"], ["Q7777", "Q525"]),
    ("science", "space",
     "NASA Artemis Mission Returns Humans to Lunar Surface After Decades",
     "First crewed moon landing since Apollo program opens new era of permanent lunar presence.",
     ["Q596", "Q525"], ["Q7777", "Q30"]),
    ("science", "space",
     "Mars Perseverance Rover Discovers Organic Molecules in Ancient Rock",
     "Biosignature candidates require sample return mission for definitive astrobiology analysis.",
     ["Q1343", "Q7777"], ["Q186447", "Q11348"]),
    ("science", "space",
     "Private Company Plans World First Commercial Space Station by 2028",
     "Module design incorporates artificial gravity rotating sections and advanced life support systems.",
     ["Q7777", "Q4830453"], ["Q596", "Q7367"]),
    ("science", "environment",
     "Breakthrough in Oceanic Plastic Recycling Using Engineered Enzymes",
     "Biochemists engineer bacterial enzymes capable of depolymerizing PET plastic in hours.",
     ["Q11438", "Q8047"], ["Q11288", "Q11348"]),
    ("science", "environment",
     "Antarctic Ice Sheet Melting Faster Than Worst Case Climate Projections",
     "Satellite data reveals acceleration that could add meters to global sea levels by century end.",
     ["Q2566", "Q11288"], ["Q7777", "Q318"]),
    ("science", "environment",
     "Coral Reef Restoration Technique Grows New Reef in Record Time",
     "Assisted evolution approach uses heat-resistant coral fragments to rebuild bleached ecosystems.",
     ["Q11288", "Q2566"], ["Q11438", "Q8047"]),
    ("science", "physics",
     "Quantum Computing Team Demonstrates Fault-Tolerant Logical Qubits",
     "Laboratory achieves quantum error correction milestone on superconducting processors.",
     ["Q413", "Q11379"], ["Q7367", "Q11660"]),
    ("science", "physics",
     "Fusion Energy Reactor Achieves Net Energy Gain for Extended Duration",
     "Plasma confinement facility sustains reaction long enough for practical power generation assessment.",
     ["Q413", "Q11379"], ["Q12705", "Q2566"]),
    ("science", "biology",
     "Synthetic Biology Creates Self-Replicating Minimal Cell from Scratch",
     "Researchers assemble living cell with just 473 genes demonstrating minimal requirements for life.",
     ["Q41710", "Q7187"], ["Q172807", "Q11348"]),
    ("science", "biology",
     "New Antibiotic Class Defeats Drug-Resistant Superbugs in Animal Models",
     "Compound targets bacterial cell wall synthesis pathway different from all existing drug classes.",
     ["Q41710", "Q11190"], ["Q8047", "Q11348"]),
]

ENTERTAINMENT_TEMPLATES = [
    ("entertainment", "movies",
     "Christopher Nolan Announces Historical Epic Production for 2026 Release",
     "Universal Studios confirms exclusive IMAX production exploring technological shifts of early 20th century.",
     ["Q43203", "Q11424"], ["Q638", "Q571"]),
    ("entertainment", "movies",
     "Marvel Cinematic Universe Announces Phase Six Lineup with New Heroes",
     "Studio reveals multi-year roadmap introducing cosmic and street level characters to franchise.",
     ["Q11424", "Q7889"], ["Q638", "Q571"]),
    ("entertainment", "movies",
     "Independent Film Wins Palme d'Or at Cannes Film Festival",
     "Low budget debut feature from first time director earns unanimous jury praise.",
     ["Q11424", "Q638"], ["Q571", "Q43203"]),
    ("entertainment", "music",
     "Grammy Awards Celebrate Global Fusion and Independent Songwriters",
     "Recording academy honors international collaborations across electronic, folk, and afrobeat genres.",
     ["Q41254", "Q638"], ["Q571", "Q11424"]),
    ("entertainment", "music",
     "Streaming Revenue Surpasses Physical Sales for First Time in Industry History",
     "Digital consumption completely dominates recorded music market transforming label economics.",
     ["Q638", "Q41254"], ["Q4830453", "Q11424"]),
    ("entertainment", "music",
     "K-Pop Group Breaks Consecutive World Records for Album Sales",
     "Global fanbase coordinates simultaneous purchasing across platforms for record breaking release.",
     ["Q638", "Q7889"], ["Q41254", "Q11424"]),
    ("entertainment", "gaming",
     "Open World RPG Sets All-Time Launch Record with 20 Million Sales",
     "Gamers praise dynamic procedural narrative system and seamless cooperative multiplayer modes.",
     ["Q7889", "Q1889"], ["Q11424", "Q7367"]),
    ("entertainment", "gaming",
     "Esports World Championship Draws 50 Million Live Concurrent Viewers",
     "Professional gaming event eclipses traditional sports finals in global viewership metrics.",
     ["Q7889", "Q349"], ["Q1889", "Q638"]),
    ("entertainment", "streaming",
     "Streaming Platforms Shift Strategy Toward Live Sports Programming",
     "Major entertainment networks invest billions in securing regional broadcasting rights.",
     ["Q11424", "Q4830453"], ["Q638", "Q2718"]),
    ("entertainment", "books",
     "Sci-Fi Novel Exploring First Contact Becomes Decade Bestseller",
     "Author blends rigorous astrophysics with human emotional drama in bestselling debut trilogy.",
     ["Q571", "Q24925"], ["Q11424", "Q638"]),
]

POLITICS_TEMPLATES = [
    ("politics", "international",
     "G7 Summit Agrees on Coordinated Response to Global Economic Challenges",
     "Leaders commit to coordinated fiscal policies and digital taxation framework for multinationals.",
     ["Q161174", "Q828"], ["Q30", "Q7278"]),
    ("politics", "international",
     "UN Security Council Passes Resolution on Artificial Intelligence Governance",
     "Member states agree framework for responsible AI development and deployment standards.",
     ["Q161174", "Q11660"], ["Q7278", "Q30"]),
    ("politics", "international",
     "NATO Expands Eastern Flank Presence Following Security Concerns",
     "Alliance agrees to permanent brigade deployments along eastern border member states.",
     ["Q1065", "Q828"], ["Q159", "Q7278"]),
    ("politics", "us-politics",
     "Congress Passes Landmark Infrastructure Investment and Climate Act",
     "Bipartisan legislation allocates record funding for clean energy and broadband expansion.",
     ["Q30", "Q7278"], ["Q12705", "Q2566"]),
    ("politics", "us-politics",
     "Supreme Court Issues Landmark Ruling on Technology and Free Speech",
     "Justices clarify First Amendment protections for social media content moderation decisions.",
     ["Q30", "Q7278"], ["Q7367", "Q6581097"]),
    ("politics", "international",
     "China-US Trade Tensions Escalate Over Semiconductor Export Controls",
     "New restrictions on advanced chip technology exports rattle global supply chains.",
     ["Q148", "Q30"], ["Q11442", "Q4830453"]),
    ("politics", "international",
     "European Union Finalizes AI Regulation Framework Setting Global Standard",
     "Comprehensive law classifies AI systems by risk level requiring mandatory conformity assessment.",
     ["Q828", "Q11660"], ["Q7278", "Q21146257"]),
]

LIFESTYLE_TEMPLATES = [
    ("lifestyle", "food",
     "Mediterranean Diet Named Healthiest Eating Pattern by Global Experts",
     "Annual review by nutrition scientists confirms plant-rich diet reduces chronic disease risk.",
     ["Q174", "Q131758"], ["Q21114", "Q11190"]),
    ("lifestyle", "food",
     "Lab-Grown Meat Receives Regulatory Approval in Key Markets",
     "Cultivated chicken and beef products reach restaurant menus after years of development.",
     ["Q174", "Q41710"], ["Q131758", "Q11190"]),
    ("lifestyle", "travel",
     "Remote Work Nomads Drive Boom in Long-Stay Tourism Markets",
     "Destination cities compete for digital workers with infrastructure and visa program improvements.",
     ["Q4830453", "Q7367"], ["Q21114", "Q131758"]),
    ("lifestyle", "travel",
     "Sustainable Aviation Fuel Adoption Accelerates Across Major Airlines",
     "Carriers commit to blending targets as production capacity for green jet fuel scales.",
     ["Q2566", "Q4830453"], ["Q12705", "Q174"]),
    ("lifestyle", "fashion",
     "Circular Fashion Movement Challenges Fast Fashion Industry Practices",
     "Resale platforms and rental services capture growing market share from traditional retailers.",
     ["Q5119", "Q4830453"], ["Q2566", "Q174"]),
    ("lifestyle", "fashion",
     "AI-Powered Personal Styling Apps Transform Wardrobe Management",
     "Machine learning models analyze body proportions and color analysis to recommend outfits.",
     ["Q11660", "Q5119"], ["Q21146257", "Q4830453"]),
]

# Collect all templates
ALL_TEMPLATES = (
    TECH_TEMPLATES * 10 +    # inflate to ~200
    SPORTS_TEMPLATES * 7 +   # ~105
    FINANCE_TEMPLATES * 8 +  # ~104
    HEALTH_TEMPLATES * 10 +  # ~100
    SCIENCE_TEMPLATES * 10 + # ~110
    ENTERTAINMENT_TEMPLATES * 10 + # ~100
    POLITICS_TEMPLATES * 7 + # ~49
    LIFESTYLE_TEMPLATES * 7  # ~42
)

# Shuffle deterministically
random.shuffle(ALL_TEMPLATES)

def vary_title(title, seed):
    rng = random.Random(seed)
    prefixes = ["Report:", "Breaking:", "Analysis:", "Update:", "Exclusive:", "Research:", "Study:", ""]
    suffix_additions = [
        " — New Report", " Amid Growing Concerns", " According to Experts",
        " As Trends Continue", " Sparks Debate", " Sets New Benchmark", ""
    ]
    prefix = rng.choice(prefixes)
    suffix = rng.choice(suffix_additions)
    if prefix:
        return f"{prefix} {title}{suffix}"
    return f"{title}{suffix}"

def vary_abstract(abstract, seed):
    rng = random.Random(seed)
    openers = [
        "New data confirms", "Industry experts report", "Researchers find", "Analysis reveals",
        "Officials announce", "Studies show", "Market observers note", "Sources indicate",
        "According to new findings,", "Experts agree that", "The latest evidence suggests",
    ]
    opener = rng.choice(openers)
    # Lowercase first char of original abstract for appending
    if abstract and abstract[0].isupper():
        abstract_mod = abstract[0].lower() + abstract[1:]
    else:
        abstract_mod = abstract
    return f"{opener} {abstract_mod}"

# Generate article list
ARTICLES = []
for idx, (cat, subcat, title, abstract, t_ents, a_ents) in enumerate(ALL_TEMPLATES):
    nid = f"N{10001 + idx}"
    varied_title = vary_title(title, seed=idx)
    varied_abstract = vary_abstract(abstract, seed=idx + 10000)
    ARTICLES.append({
        "news_id": nid,
        "category": cat,
        "subcategory": subcat,
        "title": varied_title,
        "abstract": varied_abstract,
        "url": f"https://msn.com/en-us/news/{cat}/{nid.lower()}",
        "title_entities": t_ents,
        "abstract_entities": a_ents,
    })

# Keep unique titles
seen_titles = set()
unique_articles = []
for a in ARTICLES:
    if a["title"] not in seen_titles:
        seen_titles.add(a["title"])
        unique_articles.append(a)
ARTICLES = unique_articles[:1000]  # cap at 1000

print(f"[Generator] Prepared {len(ARTICLES)} unique articles")

# ───────────────────────────────────────────────────────────────────────────────
# User Persona Definitions
# ───────────────────────────────────────────────────────────────────────────────
CAT_BUCKETS = {}
for a in ARTICLES:
    CAT_BUCKETS.setdefault(a["category"], []).append(a["news_id"])

# Persona: (primary_cats, secondary_cats, weight_primary)
PERSONAS = [
    (["technology"], ["science", "finance"], 0.7),
    (["sports"], ["lifestyle", "entertainment"], 0.75),
    (["finance"], ["politics", "technology"], 0.65),
    (["health"], ["lifestyle", "science"], 0.7),
    (["science"], ["technology", "health"], 0.65),
    (["entertainment"], ["lifestyle", "sports"], 0.7),
    (["politics"], ["finance", "international"], 0.6),
    (["lifestyle"], ["health", "entertainment"], 0.65),
    (["technology", "science"], ["finance", "health"], 0.6),
    (["sports", "entertainment"], ["lifestyle", "politics"], 0.65),
]

def get_persona_articles(persona, n_history=8, n_impressions=10, rng=None):
    if rng is None:
        rng = random
    primary_cats, secondary_cats, w = persona
    
    all_primary = []
    for c in primary_cats:
        all_primary.extend(CAT_BUCKETS.get(c, []))
    all_secondary = []
    for c in secondary_cats:
        all_secondary.extend(CAT_BUCKETS.get(c, []))
    all_other = [a["news_id"] for a in ARTICLES
                 if a["category"] not in primary_cats + secondary_cats]
    
    if not all_primary:
        all_primary = [a["news_id"] for a in ARTICLES]
    
    # Build history
    history = []
    for _ in range(n_history):
        r = rng.random()
        if r < w and all_primary:
            history.append(rng.choice(all_primary))
        elif r < w + 0.2 and all_secondary:
            history.append(rng.choice(all_secondary))
        elif all_other:
            history.append(rng.choice(all_other))
    history = list(dict.fromkeys(history))[:n_history]  # unique, ordered
    
    # Build impressions: some positive clicks, some negatives
    n_pos = rng.randint(1, 3)
    n_neg = n_impressions - n_pos
    
    pos_pool = [n for n in all_primary if n not in history]
    neg_pool = [a["news_id"] for a in ARTICLES if a["news_id"] not in history]
    
    if not pos_pool:
        pos_pool = all_primary[:]
    
    positives = rng.sample(pos_pool, k=min(n_pos, len(pos_pool)))
    neg_pool = [n for n in neg_pool if n not in positives]
    negatives = rng.sample(neg_pool, k=min(n_neg, len(neg_pool)))
    
    candidates = [(n, 1) for n in positives] + [(n, 0) for n in negatives]
    rng.shuffle(candidates)
    
    return history, candidates


def generate_mind_small_dataset(output_dir, n_train_users=400, n_dev_users=100):
    os.makedirs(output_dir, exist_ok=True)

    for split, n_users, imp_offset in [("train", n_train_users, 1), ("dev", n_dev_users, 100000)]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # Write news.tsv
        news_path = os.path.join(split_dir, "news.tsv")
        with open(news_path, "w", encoding="utf-8") as f:
            for art in ARTICLES:
                t_ents_json = json.dumps([make_entity(e) for e in art["title_entities"]])
                a_ents_json = json.dumps([make_entity(e) for e in art["abstract_entities"]])
                row = "\t".join([
                    art["news_id"],
                    art["category"],
                    art["subcategory"],
                    art["title"],
                    art["abstract"],
                    art["url"],
                    t_ents_json,
                    a_ents_json,
                ]) + "\n"
                f.write(row)
        print(f"[{split.upper()}] Wrote {len(ARTICLES)} articles → {news_path}")

        # Write behaviors.tsv
        beh_path = os.path.join(split_dir, "behaviors.tsv")
        base_time = datetime(2024, 11, 1, 8, 0, 0)
        
        with open(beh_path, "w", encoding="utf-8") as f:
            imp_id = imp_offset
            for u_idx in range(1, n_users + 1):
                user_id = f"U{u_idx:04d}" if split == "train" else f"U{5000 + u_idx:04d}"
                persona = PERSONAS[(u_idx - 1) % len(PERSONAS)]
                rng = random.Random(u_idx * 1337 + (0 if split == "train" else 9999))
                
                # Each user has 1-4 impression sessions
                n_sessions = rng.randint(1, 4)
                for s in range(n_sessions):
                    history, candidates = get_persona_articles(persona, n_history=rng.randint(3, 15), n_impressions=rng.randint(5, 15), rng=rng)
                    
                    history_str = " ".join(history)
                    impressions_str = " ".join([f"{n}-{lbl}" for n, lbl in candidates])
                    
                    ts = base_time + timedelta(hours=u_idx * 2 + s * 6, minutes=rng.randint(0, 59))
                    ts_str = ts.strftime("%-m/%-d/%Y %-I:%M:%S %p") if os.name != "nt" else ts.strftime("%#m/%#d/%Y %#I:%M:%S %p")
                    
                    row = "\t".join([str(imp_id), user_id, ts_str, history_str, impressions_str]) + "\n"
                    f.write(row)
                    imp_id += 1

        actual_imps = imp_id - imp_offset
        print(f"[{split.upper()}] Wrote {actual_imps:,} impression rows for {n_users} users → {beh_path}")

    print(f"\n✓ MIND-small dataset generated in: {output_dir}")
    print(f"  Articles: {len(ARTICLES):,}")
    print(f"  Categories: {len(CAT_BUCKETS)}")
    print(f"  Entity types: {len(set(e for a in ARTICLES for e in a['title_entities'] + a['abstract_entities']))}")


if __name__ == "__main__":
    out = os.path.join(DATA_DIR)
    generate_mind_small_dataset(out, n_train_users=400, n_dev_users=100)

