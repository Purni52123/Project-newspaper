"""
Sample MIND Dataset Generator
Generates a realistic small-scale dataset matching the Microsoft MIND dataset format.
Allows testing the complete pipeline, training, API, and web interface without large downloads.
"""

import os
import json
import random

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_ARTICLES = [
    # Technology / AI
    ("N1001", "technology", "artificial-intelligence", "DeepSeek and OpenAI Announce Next Generation Reasoning Models", 
     "New breakthroughs in reinforcement learning and neural chain of thought push artificial intelligence forward.", 
     [{"WikidataId": "Q11660", "Label": "Artificial Intelligence"}, {"WikidataId": "Q65640329", "Label": "OpenAI"}]),
    ("N1002", "technology", "gadgets", "Apple Unveils M4 Pro Silicon Chip for High-Performance Workstations", 
     "Apple has launched its latest silicon architecture offering unprecedented hardware acceleration for AI tasks.", 
     [{"WikidataId": "Q312", "Label": "Apple Inc."}, {"WikidataId": "Q5300", "Label": "Silicon"}]),
    ("N1003", "technology", "cybersecurity", "Global Cyber Attack Targets Cloud Infrastructure Services", 
     "Security researchers identify zero-day vulnerabilities in containerized microservices across leading providers.", 
     [{"WikidataId": "Q8078", "Label": "Cybersecurity"}, {"WikidataId": "Q483247", "Label": "Cloud Computing"}]),
    ("N1004", "technology", "artificial-intelligence", "Graph Neural Networks Revolutionize Drug Discovery", 
     "Biotech researchers use geometric deep learning and graph neural networks to simulate molecular bonds.", 
     [{"WikidataId": "Q11660", "Label": "Artificial Intelligence"}, {"WikidataId": "Q11190", "Label": "Medicine"}]),
    ("N1005", "technology", "robotics", "Humanoid Robots Enter Automated Logistics Warehouses", 
     "Autonomous bipedal robots are deployed to assist human workers in packing and sorting shipping parcels.", 
     [{"WikidataId": "Q11012", "Label": "Robotics"}, {"WikidataId": "Q177777", "Label": "Logistics"}]),
    
    # Sports
    ("N2001", "sports", "football", "Real Madrid Secures Dramatic Champions League Victory", 
     "A stoppage time strike seals a stunning comeback victory in European football quarterfinals.", 
     [{"WikidataId": "Q8682", "Label": "Real Madrid"}, {"WikidataId": "Q2718", "Label": "Football"}]),
    ("N2002", "sports", "basketball", "Lakers Stage Fourth Quarter Comeback Against Golden State", 
     "LeBron James and Anthony Davis combine for 65 points in a thrilling Western Conference showdown.", 
     [{"WikidataId": "Q121776", "Label": "Los Angeles Lakers"}, {"WikidataId": "Q5372", "Label": "Basketball"}]),
    ("N2003", "sports", "cricket", "India Clinches Thrilling T20 World Cup Final Against South Africa", 
     "Sensational bowling in the death overs leads team India to a historic world cup championship triumph.", 
     [{"WikidataId": "Q9842", "Label": "Cricket"}, {"WikidataId": "Q668", "Label": "India"}]),
    ("N2004", "sports", "formula1", "Ferrari Dominates Italian Grand Prix With Masterful Strategy", 
     "Charles Leclerc holds off McLaren drivers to claim an emotional home victory at Monza circuit.", 
     [{"WikidataId": "Q171321", "Label": "Formula One"}, {"WikidataId": "Q169898", "Label": "Ferrari"}]),
    ("N2005", "sports", "tennis", "Carlos Alcaraz Defends Grand Slam Title in Five Set Marathon", 
     "The Spanish prodigy outlasts his opponent in a grueling five-hour final on court central.", 
     [{"WikidataId": "Q850239", "Label": "Tennis"}, {"WikidataId": "Q62024195", "Label": "Carlos Alcaraz"}]),
     
    # Business & Finance
    ("N3001", "finance", "markets", "Federal Reserve Holds Interest Rates Steady Amid Economic Resilience", 
     "Central bank chair indicates monetary policy is well positioned to control inflation while maintaining growth.", 
     [{"WikidataId": "Q53536", "Label": "Federal Reserve"}, {"WikidataId": "Q8142", "Label": "Economy"}]),
    ("N3002", "finance", "crypto", "Bitcoin Surges Past Milestone High as Institutional Inflows Accelerate", 
     "Surging exchange-traded fund activity and digital asset regulatory clarity drive market momentum.", 
     [{"WikidataId": "Q131723", "Label": "Bitcoin"}, {"WikidataId": "Q13479982", "Label": "Cryptocurrency"}]),
    ("N3003", "finance", "stocks", "Semiconductor Stocks Rally Following Strong Global Cloud Demand", 
     "Chipmakers surge after industry earnings demonstrate robust corporate capital expenditure on servers.", 
     [{"WikidataId": "Q11442", "Label": "Semiconductor"}, {"WikidataId": "Q11691", "Label": "Stock Market"}]),
    ("N3004", "finance", "real-estate", "Commercial Real Estate Adapts to Hybrid Work Trends", 
     "Office building landlords renovate properties into multi-use spaces with wellness and flexible amenities.", 
     [{"WikidataId": "Q68947", "Label": "Real Estate"}, {"WikidataId": "Q4830453", "Label": "Business"}]),
    ("N3005", "finance", "startup", "Venture Capital Focuses on Sustainable Energy and Climate Tech", 
     "Clean energy startups draw record early-stage investment into next-generation sodium battery technology.", 
     [{"WikidataId": "Q217409", "Label": "Venture Capital"}, {"WikidataId": "Q12705", "Label": "Clean Energy"}]),
     
    # Health & Science
    ("N4001", "health", "medicine", "New mRNA Vaccine Shows Promise in Targeted Cancer Immunotherapy", 
     "Clinical trials report durable immune responses against difficult-to-treat solid tumor malignancies.", 
     [{"WikidataId": "Q12111", "Label": "Cancer"}, {"WikidataId": "Q131765", "Label": "mRNA"}]),
    ("N4002", "health", "wellness", "Intermittent Fasting and Circadian Rhythms: What the Science Shows", 
     "Nutritional scientists study how meal timing impacts metabolic markers and cellular repair mechanisms.", 
     [{"WikidataId": "Q131758", "Label": "Nutrition"}, {"WikidataId": "Q21114", "Label": "Health"}]),
    ("N4003", "science", "space", "James Webb Space Telescope Discovers Ancient Galaxy Near Cosmic Dawn", 
     "Astronomers observe stellar formations dating back to just a few hundred million years after the Big Bang.", 
     [{"WikidataId": "Q186447", "Label": "James Webb Space Telescope"}, {"WikidataId": "Q318", "Label": "Galaxy"}]),
    ("N4004", "science", "environment", "Breakthrough in Oceanic Plastic Recycling Using Engineered Enzymes", 
     "Biochemists engineer bacterial enzymes capable of depolymerizing PET plastic in hours at room temperature.", 
     [{"WikidataId": "Q11438", "Label": "Plastic"}, {"WikidataId": "Q8047", "Label": "Enzyme"}]),
    ("N4005", "science", "physics", "Quantum Computing Team Demonstrates Fault-Tolerant Logical Qubits", 
     "Experimental physics laboratory achieves quantum error correction milestone on superconducting processors.", 
     [{"WikidataId": "Q11190", "Label": "Quantum Computing"}, {"WikidataId": "Q413", "Label": "Physics"}]),

    # Entertainment
    ("N5001", "entertainment", "movies", "Director Christopher Nolan Announces Historical Epic for 2026", 
     "Universal Studios confirms exclusive IMAX production exploring technological shifts of early 20th century.", 
     [{"WikidataId": "Q43203", "Label": "Christopher Nolan"}, {"WikidataId": "Q11424", "Label": "Film"}]),
    ("N5002", "entertainment", "music", "Grammy Awards Celebrate Global Fusion and Indie Songwriters", 
     "The recording academy honors international collaborations across electronic, folk, and afrobeat genres.", 
     [{"WikidataId": "Q41254", "Label": "Grammy Award"}, {"WikidataId": "Q638", "Label": "Music"}]),
    ("N5003", "entertainment", "gaming", "Next-Generation Open World RPG Sets All-Time Launch Record", 
     "Gamers praise the dynamic procedural narrative system and seamless cooperative multiplayer modes.", 
     [{"WikidataId": "Q7889", "Label": "Video Game"}, {"WikidataId": "Q1889", "Label": "Gaming"}]),
    ("N5004", "entertainment", "books", "Bestselling Sci-Fi Novel Explores First Contact and Multiverse Theories", 
     "Author blends rigorous theoretical astrophysics with human emotional drama in bestselling debut trilogy.", 
     [{"WikidataId": "Q571", "Label": "Book"}, {"WikidataId": "Q24925", "Label": "Science Fiction"}]),
    ("N5005", "entertainment", "streaming", "Streaming Platforms Shift Toward Curated Live Sports Programming", 
     "Major entertainment networks invest billions in securing regional broadcasting rights for live athletic matches.", 
     [{"WikidataId": "Q11424", "Label": "Media"}, {"WikidataId": "Q2718", "Label": "Sports"}]),
]

def generate_sample_dataset():
    random.seed(42)
    
    for split in ["train", "dev"]:
        dest_folder = os.path.join(DATA_DIR, split)
        os.makedirs(dest_folder, exist_ok=True)
        
        # Write news.tsv
        news_path = os.path.join(dest_folder, "news.tsv")
        with open(news_path, "w", encoding="utf-8") as f:
            for nid, cat, subcat, title, abstract, entities in SAMPLE_ARTICLES:
                t_entities = json.dumps(entities[:1])
                a_entities = json.dumps(entities[1:] if len(entities) > 1 else entities)
                url = f"https://assets.msn.com/news/{nid}"
                row = f"{nid}\t{cat}\t{subcat}\t{title}\t{abstract}\t{url}\t{t_entities}\t{a_entities}\n"
                f.write(row)
        
        # Generate behaviors.tsv
        # 30 users with distinct reading personas
        behaviors_path = os.path.join(dest_folder, "behaviors.tsv")
        news_ids = [a[0] for a in SAMPLE_ARTICLES]
        
        # Categories mapping
        tech_ids = [a[0] for a in SAMPLE_ARTICLES if a[1] == "technology"]
        sports_ids = [a[0] for a in SAMPLE_ARTICLES if a[1] == "sports"]
        fin_ids = [a[0] for a in SAMPLE_ARTICLES if a[1] == "finance"]
        sci_ids = [a[0] for a in SAMPLE_ARTICLES if a[1] in ("health", "science")]
        ent_ids = [a[0] for a in SAMPLE_ARTICLES if a[1] == "entertainment"]
        
        clusters = [tech_ids, sports_ids, fin_ids, sci_ids, ent_ids]
        
        with open(behaviors_path, "w", encoding="utf-8") as f:
            imp_id = 1 if split == "train" else 1000
            for u_idx in range(1, 31):
                user_id = f"U{u_idx:04d}"
                # Primary interest cluster
                fav_cluster = clusters[(u_idx - 1) % len(clusters)]
                # History (2 to 4 articles from favorite cluster)
                hist = random.sample(fav_cluster, k=min(3, len(fav_cluster)))
                history_str = " ".join(hist)
                
                # Impressions: candidate news with 1 (clicked) and 0 (not clicked)
                # 1 clicked from favorite or adjacent cluster, 4 unclicked
                pos = random.choice([nid for nid in fav_cluster if nid not in hist] or fav_cluster)
                neg_pool = [nid for nid in news_ids if nid != pos and nid not in hist]
                negs = random.sample(neg_pool, k=min(4, len(neg_pool)))
                
                candidates = [(pos, 1)] + [(n, 0) for n in negs]
                random.shuffle(candidates)
                impressions_str = " ".join([f"{n}-{lbl}" for n, lbl in candidates])
                
                timestamp = f"11/11/2026 09:{u_idx:02d}:00 AM"
                row = f"{imp_id}\t{user_id}\t{timestamp}\t{history_str}\t{impressions_str}\n"
                f.write(row)
                imp_id += 1

        print(f"[{split.upper()}] Generated sample data in {dest_folder}: {len(SAMPLE_ARTICLES)} articles, 30 user behaviors.")

if __name__ == "__main__":
    generate_sample_dataset()

