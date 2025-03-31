from datetime import datetime
import os
import random
import json
import numpy as np
from PIL import Image  # Import PIL for image manipulation

def load_probabilities(file_path):
    """Load probabilities from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def invert_image(img):
    # Split into bands
    r, g, b, a = img.split()
    
    # Invert RGB channels only, preserve alpha
    r_inv = Image.fromarray(255 - np.array(r))
    g_inv = Image.fromarray(255 - np.array(g))
    b_inv = Image.fromarray(255 - np.array(b))
    
    # Merge channels back and return
    return Image.merge('RGBA', (r_inv, g_inv, b_inv, a))

def generate_layer_choices(probabilities, num_images):
    """Generate layer choices based on probabilities."""
    layer_choices = {}
    for dir, prob in probabilities.items():
        #sample = np.random.multinomial(num_images, prob)
        #probs_list = [i for i, s in enumerate(sample) for _ in range(s)]
        layer_choices[dir] = [random.randint(0, len(prob)-1) for _ in range(num_images)]
    return layer_choices

def combine_layers_to_image(layers, metadata_path, directories, current_directory, output_path):
    """Combine layers into a single image."""
    base_image = Image.new("RGBA", (1000, 1248), (0, 0, 0, 0))  # Create a new transparent image
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    for i, (dir, layer) in enumerate(zip(directories, layers)):
        layer_path = os.path.join(current_directory, dir, f"{layer}.png")
        try:
            with Image.open(layer_path) as img_layer:
                img_layer = img_layer.convert("RGBA")  # Ensure the layer is in RGBA mode
                if random.random() < 0.05:
                    metadata["attributes"][i]["value"] = f'foil-{metadata["attributes"][i]["value"]}'
                    img_layer = invert_image(img_layer)
                img_layer = img_layer.resize(base_image.size)
                base_image = Image.alpha_composite(base_image, img_layer)
                #base_image.paste(img_layer, (0, 0))
        except FileNotFoundError:
            print(f"Warning: File not found: {layer_path}")
    base_image.save(output_path)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

names = [{
    "0": "avalanche",
    "1": "er-visit",
    "2": "hot-dog",
    "3": "snail-race",
    "4": "beach-day",
    "5": "mountain-climb", 
    "6": "city-lights",
    "7": "forest-trail",
    "8": "desert-sunset",
    "9": "ocean-waves",
    "10": "space-stars",
    "11": "jungle-path", 
    "12": "snow-storm",
    "13": "desert-dunes",
    "14": "mountain-peaks",
    "15": "ocean-waves",
    "16": "volcanic-caves",
    "17": "crystal-mines",
    "18": "cloud-city",
    "19": "forest-ruins",
    "20": "ice-caverns",
    "21": "swamp-lands",
    "22": "ancient-temple",
    "23": "floating-islands",
    "24": "mystic-gardens",
    "25": "cyber-grid",
    "26": "haunted-mansion",
    "27": "sky-fortress",
    "28": "deep-ocean",
    "29": "lava-fields",
    "30": "quantum-realm",
    "31": "astral-plane",
    "32": "shadow-realm",
    "33": "dream-world",
    "34": "time-vortex",
    "35": "cosmic-void",
    "36": "dragon-peaks",
    "37": "fairy-grove",
    "38": "thunder-plains",
    "39": "crystal-caves",
    "40": "mechanical-city",
    "41": "spirit-woods",
    "42": "void-dimension",
    "43": "meteor-field",
    "44": "rainbow-roads",
    "45": "parallel-world",
    "46": "gravity-wells",
    "47": "nebula-clouds",
    "48": "bone-valley",
    "49": "plasma-storm",
    "50": "digital-maze",
    "51": "endless-desert",
    "52": "frost-peaks",
    "53": "mushroom-forest",
    "54": "electric-fields",
    "55": "nightmare-realm",
    "56": "coral-reefs",
    "57": "toxic-wastes",
    "58": "alien-hive",
    "59": "quantum-city",
    "60": "mirror-dimension",
    "61": "steam-works",
    "62": "crystal-peaks",
    "63": "dark-matter",
    "64": "light-realm",
    "65": "time-ruins",
    "66": "void-gates",
    "67": "cosmic-sea",
    "68": "dream-spiral",
    "69": "shadow-peaks",
    "70": "star-forge",
    "71": "energy-grid",
    "72": "crystal-core",
    "73": "omega-realm",
    "74": "shadow-gate",
    "75": "frozen-peak",
    "76": "dragon-shrine",
    "77": "mystic-oasis", 
    "78": "storm-citadel",
    "79": "ancient-tomb",
    "80": "solar-temple",
    "81": "void-chamber",
    "82": "astral-well",
    "83": "phoenix-spire",
    "84": "cosmic-altar"
    },
    {
    "0": "moving-shadow",
    "1": "alchemical-frog",
    "2": "quantum-butterfly",
    "3": "holographic-dragon", 
    "4": "sentient-cloudscape",
    "5": "neon-jellyfish", 
    "6": "robotic-tentacle",
    "7": "temporal-lightning", 
    "8": "living-fractal",
    "9": "interdimensional-prism", 
    "10": "cybernetic-webbing", 
    "11": "magical-smoke", 
    "12": "alien-crystal", 
    "13": "dimensional-mist", 
    "14": "quantum-sparkle", 
    "15": "telepathic-pulse", 
    "16": "time-ripple", 
    "17": "ghostly-membrane", 
    "18": "cosmic-lattice", 
    "19": "synthetic-web", 
    "20": "plasma-veil", 
    "21": "digital-glitch", 
    "22": "ethereal-flame", 
    "23": "spectral-lens", 
    "24": "energy-matrix", 
    "25": "morphing-grid", 
    "26": "neural-network", 
    "27": "quantum-foam", 
    "28": "holographic-haze", 
    "29": "dark-matter", 
    "30": "void-tentacle", 
    "31": "psychic-filter", 
    "32": "temporal-distortion", 
    "33": "antimatter-veil", 
    "34": "sentient-circuit", 
    "35": "living-data", 
    "36": "dimensional-static", 
    "37": "quantum-spray", 
    "38": "cosmic-debris", 
    "39": "energy-web", 
    "40": "digital-shadow", 
    "41": "spectral-pulse", 
    "42": "time-membrane", 
    "43": "neural-mist", 
    "44": "holographic-noise"
    },
    {
    "0": "永不眠", # never-sleep
    "1": "精神分裂", # schizophrenia
    "2": "辞职", # quit-job
    "3": "心脏病发作", # heart-attack
    "4": "梦中梦", # dream-in-dream
    "5": "存在危机", # existential-crisis 
    "6": "午夜恐慌", # midnight-panic
    "7": "量子困惑", # quantum-confusion
    "8": "数字崩溃", # digital-breakdown
    "9": "宇宙w绝望", # cosmic-despair
    "10": "现实故障", # reality-glitch
    "11": "时间眩晕", # temporal-vertigo
    "12": "神经过载", # neural-overload
    "13": "无限循环", # infinite-loop
    "14": "形而上学崩溃", # metaysical-meltdown
    "15": "自我消解", # ego-dissolution
    "16": "业力内h爆", # karmic-implosion
    "17": "精神龙卷风", # spiritual-torna
    "18": "意识崩d溃", # consciousness-crash
    "19": "维度扭曲", # dimensional-warp
    "20": "心灵内爆", # psychic-implosion
    "21": "情绪海啸", # emotional-tsunami
    "22": "宇宙眩晕", # cosmic-vertigo
    "23": "量子崩溃", # quantum-breakdown
    "24": "存在虚空", # existential-void
    "25": "形而上学恐慌", # metaphysical-panic
    "26": "神经爆炸", # neural-explosion
    "27": "现实熔毁", # reality-meltdown
    "28": "时空解w体", # temporal-disintegration
    "29": "无限递归", # infinite-recursion
    "30": "数字精神病" # digital-psychosis
    },
    {
    "0": "dale-earnhardt-jr", # NASCAR driver
    "1": "usain-bolt", # Fastest sprinter
    "2": "genghis-khan", # Rapid empire expansion
    "3": "nick-land", # Accelerationist philosopher
    "4": "gottfried-leibniz", # Early accelerationist thinking
    "5": "muhammad-ali", # Boxing speed
    "6": "bruce-lee", # Martial arts speed
    "7": "carl-lewis", # Olympic sprinter
    "8": "ayrton-senna", # F1 driver
    "9": "jesse-owens", # Olympic sprinter
    "10": "michael-phelps", # Swimming speed
    "11": "paul-virilio", # Speed theorist
    "12": "marinetti", # Futurist movement
    "13": "sun-tzu", # Military speed
    "14": "napoleon-bonaparte", # Military campaigns
    "15": "alexander-great", # Rapid conquest
    "16": "julius-caesar", # Roman expansion
    "17": "ramanujan", # Mathematical speed
    "18": "john-von-neumann", # Fast computation
    "19": "alan-turing", # Computer pioneer
    "20": "einstein", # Speed of light
    "21": "edison", # Rapid innovation
    "22": "tesla", # Electric speed
    "23": "wright-brothers", # Flight pioneers
    "24": "chuck-yeager", # Sound barrier
    "25": "yuri-gagarin", # Space speed
    "26": "neil-armstrong", # Space travel
    "27": "richard-feynman", # Physics speed
    "28": "heraclitus", # Flow philosophy
    "29": "deleuze", # Accelerationist influence
    "30": "guattari", # Accelerationist thinking
    "31": "friedrich-nietzsche", # Will to power
    "32": "karl-marx", # Historical acceleration
    "33": "mao-zedong", # Rapid industrialization
    "34": "deng-xiaoping", # Economic acceleration
    "35": "henry-ford", # Industrial speed
    "36": "steve-jobs", # Tech acceleration
    "37": "bill-gates", # Software speed
    "38": "mark-zuckerberg", # Social acceleration
    "39": "elon-musk", # Transport/space speed
    "40": "jeff-bezos", # Business acceleration
    "41": "jack-ma", # Asian tech speed
    "42": "larry-page", # Search acceleration
    "43": "sergey-brin", # Info acceleration
    "44": "ada-lovelace", # Computing pioneer
    "45": "grace-hopper", # Programming speed
    "46": "claude-shannon", # Information theory
    "47": "norbert-wiener", # Cybernetics
    "48": "marshall-mcluhan", # Media acceleration
    "49": "ray-kurzweil", # Singularity theory
    "50": "max-planck", # Quantum speed
    "51": "werner-heisenberg", # Physics speed
    "52": "stephen-hawking", # Space-time theory
    "53": "roger-penrose", # Physics acceleration
    "54": "paul-dirac", # Quantum pioneer
    "55": "richard-dawkins", # Evolution speed
    "56": "james-watson", # DNA discovery
    "57": "francis-crick", # Genetic code
    "58": "craig-venter", # Genome speed
    "59": "tim-berners-lee", # Web pioneer
    "60": "vint-cerf", # Internet pioneer
    "61": "robert-kahn", # Network speed
    "62": "donald-knuth", # Algorithm speed
    "63": "linus-torvalds", # Linux speed
    "64": "dennis-ritchie", # C language
    "65": "ken-thompson", # Unix speed
    "66": "alex-jones", # Unix speed
    "67": "iron-chef-chinese",
    "68": "james-gosling", # Java speed
    "69": "brendan-eich", # JavaScript pioneer
    "70": "anders-hejlsberg", # C# acceleration
    "71": "yukihiro-matsumoto", # Ruby creator
    "72": "rasmus-lerdorf", # PHP speed
    "73": "larry-wall", # Perl creator
    "74": "martin-odersky", # Scala speed
    },
    {
    "0": "sleep-walker",
    "1": "undead-snail",
    "2": "middle-finger",
    "3": "beaver",
    "4": "rasta-rat",
    "5": "doge-wow",
    "6": "grumpy-cat",
    "7": "nyan-cat",
    "8": "pepe-frog",
    "9": "dat-boi",
    "10": "harambe-gorilla",
    "11": "distracted-boyfriend",
    "12": "hide-the-pain-harold",
    "13": "chad-face",
    "14": "wojak-cry",
    "15": "stonks-man",
    "16": "surprised-pikachu",
    "17": "this-is-fine-dog",
    "18": "galaxy-brain",
    "19": "drake-hotline",
    "20": "woman-yelling-cat",
    "21": "moths-lamp",
    "22": "ugandan-knuckles",
    "23": "dab-squidward",
    "24": "spongebob-mocking",
    "25": "rickroll-astley",
    "26": "loss-comic",
    "27": "disaster-girl",
    "28": "success-kid",
    "29": "bad-luck-brian",
    "30": "overly-attached-girlfriend",
    "31": "one-does-not-simply",
    "32": "y-u-no-guy",
    "33": "trollface-classic",
    "34": "forever-alone",
    "35": "dolan-duck",
    "36": "giga-chad",
    "37": "rage-guy",
    "38": "philosoraptor",
    "39": "scumbag-steve",
    "40": "good-guy-greg",
    "41": "socially-awkward-penguin",
    "42": "doge-cheems",
    "43": "buff-doge",
    "44": "wide-putin",
    "45": "spongegar",
    "46": "caveman-spongebob",
    "47": "handsome-squidward",
    "48": "big-chungus",
    "49": "stonks-down",
    "50": "not-stonks",
    "51": "mega-stonks",
    "52": "mr-incredible-uncanny",
    "53": "soyjak-pointing",
    "54": "amogus-sus",
    "55": "bonk-doge",
    "56": "horny-jail",
    "57": "trade-offer",
    "58": "jimmy-neutron",
    "59": "monkey-puppet",
    "60": "sweating-towel-guy",
    "61": "confused-math-lady",
    "62": "ancient-aliens-guy",
    "63": "dwayne-eyebrow",
    "64": "side-eye-chloe",
    "65": "crying-cat",
    "66": "spiderman-pointing",
    "67": "expanding-brain",
    "68": "coffin-dance",
    "69": "nice-meme",
    "70": "stonks-helth",
    "71": "panik-kalm",
    "72": "always-has-been",
    "73": "chad-yes",
    "74": "sigma-grindset",
    "75": "no-maidens",
    "76": "emotional-damage",
    "77": "gigachad-black-white",
    "78": "cat-jam",
    "79": "vibing-cat",
    "80": "aight-imma-head-out",
    "81": "we-live-in-society",
    "82": "thomas-had-seen",
    "83": "sad-pablo",
    "84": "confused-unga-bunga",
    "85": "math-confusion",
    "86": "mr-incredible-becoming-uncanny",
    "87": "troll-despair",
    "88": "skeleton-forgor",
    "89": "moyai-statue",
    "90": "walter-white-falling",
    "91": "jetstream-sam",
    "92": "standing-here",
    "93": "vine-boom",
    "94": "emotional-damage",
    "95": "gigachad-smile",
    "96": "waltuh-white",
    "97": "jesse-we-need",
    "98": "patrick-wallet",
    "99": "buzz-everywhere",
    "100": "everywhere-at-end",
    "101": "metal-pipe",
    "102": "pipe-bomb",
    "103": "gordon-ramsay-lamb",
    "104": "doom-guy-rage",
    "105": "heavy-dead",
    "106": "engineer-gaming",
    "107": "spamton-deal",
    "108": "sans-undertale",
    "109": "omori-stairs",
    "110": "hello-mario",
    "111": "super-idol",
    "112": "bing-chilling",
    "113": "social-credit",
    "114": "backrooms-entity",
    "115": "morbius-morb",
    "116": "its-morbin-time",
    "117": "quandale-dingle",
    "118": "goofy-ahh",
    "119": "skill-issue",
    "120": "jerma-sus",
    "121": "mega-pog",
    "122": "omega-lul",
    "123": "super-cap",
    "124": "based-take",
    "125": "no-shot",
    "126": "touch-grass",
    "127": "copium-max",
    "128": "omega-cringe",
    "129": "actual-kekw"
    },
    {
    "0": "velocity",
    "1": "acceleration",
    "2": "momentum",
    "3": "hypersonic",
    "4": "supersonic",
    "5": "lightspeed",
    "6": "warpspeed",
    "7": "quantum-leap",
    "8": "hyperspace",
    "9": "slipstream",
    "10": "mach-speed",
    "11": "time-warp",
    "12": "thrust",
    "13": "propulsion",
    "14": "kinetic",
    "15": "vector",
    "16": "trajectory",
    "17": "inertia",
    "18": "friction",
    "19": "impulse",
    "20": "force",
    "21": "rapid",
    "22": "swift",
    "23": "hurtle",
    "24": "dash",
    "25": "berzerk",
    "26": "crush",
    "27": "burst",
    "28": "zap",
    "29": "zoom",
    "30": "bolt",
    "31": "flash",
    "32": "race",
    "33": "sprint",
    "34": "boost",
    "35": "hustle",
    "36": "charge",
    "37": "rush",
    "38": "surge",
    "39": "blast"
    },
    {
    "0": "scribble_scrabble",
    "1": "milky",
    "2": "top-text",
    "3": "cloud-burst",
    "4": "paint-splatter",
    "5": "ink-blot",
    "6": "graffiti-tag",
    "7": "glitch-static",
    "8": "neon-glow",
    "9": "pixel-noise",
    "10": "lightning-flash",
    "11": "smoke-wisp",
    "12": "rainbow-streak",
    "13": "geometric-pattern",
    "14": "dot-matrix",
    "15": "wave-lines",
    "16": "circuit-traces",
    "17": "brush-stroke",
    "18": "star-burst",
    "19": "sparkle-scatter",
    "20": "bubble-float",
    "21": "gradient-fade",
    "22": "abstract-swirl",
    "23": "digital-rain",
    "24": "plasma-wave",
    "25": "hexagon-grid",
    "26": "triangle-mesh",
    "27": "binary-stream",
    "28": "fire-ember",
    "29": "water-ripple",
    "30": "electric-arc",
    "31": "fractal-pattern",
    "32": "vortex-spiral",
    "33": "crystal-shard",
    "34": "retro-scan",
    "35": "mosaic-blur",
    "36": "matrix-code",
    "37": "cyber-pulse"
    },
    {
    "0": "rat-clown",
    "1": "tiddly-winks",
    "2": "foreign-lobbyist",
    "3": "crypto-bro",
    "4": "meme-lord",
    "5": "dank-dealer",
    "6": "troll-face",
    "7": "pixel-punk",
    "8": "glitch-ghost",
    "9": "hack-master",
    "10": "byte-bandit",
    "11": "cyber-monk",
    "12": "data-demon",
    "13": "neon-ninja",
    "14": "vapor-viper",
    "15": "digital-drifter",
    "16": "matrix-monk",
    "17": "binary-banshee",
    "18": "code-cultist",
    "19": "net-nomad",
    "20": "web-wraith",
    "21": "tech-templar",
    "22": "spam-specter",
    "23": "virus-vagrant",
    "24": "server-shade",
    "25": "proxy-phantom",
    "26": "bug-baron",
    "27": "glitch-gremlin",
    "28": "static-spirit",
    "29": "cache-creeper",
    "30": "ram-reaper",
    "31": "bit-buccaneer",
    "32": "pixel-pirate",
    "33": "cyber-sorcerer"
    },
    {
    "0": "snail-cross",
    "1": "forward-slash",
    "2": "backward-slash",
    "3": "plus",
    "4": "asterisk",
    "5": "hash-tag",
    "6": "dollar-sign",
    "7": "percent-mark",
    "8": "ampersand",
    "9": "at-symbol",
    "10": "exclamation",
    "11": "question-mark",
    "12": "",
    "13": "",
    "14": "",
    "15": "",
    "16": "",
    "17": "",
    "18": "",
    "19": "",
    "20": "",
    "21": "",
    "22": "",
    "23": ""
    },
    {
    "0": "pink-blood",
    "1": "floating-flower",
    "2": "neon-drip",
    "3": "rainbow-trail",
    "4": "glitter-burst",
    "5": "pixel-dust",
    "6": "static-fuzz",
    "7": "cyber-spark",
    "8": "digital-aura",
    "9": "glitch-halo",
    "10": "matrix-rain",
    "11": "",
    "12": "",
    "13": "",
    "14": "",
    "15": "",
    "16": "",
    "17": "",
    "18": "",
    "19": "",
    "20": "",
    "21": ""
    },
   {
    "0": "fashion-deals-store", 
    "1": "tech-gadget-shop",
    "2": "home-decor-outlet",
    "3": "pet-supplies-market",
    "4": "sports-gear-zone",
    "5": "garden-center-store",
    "6": "kitchen-essentials-shop",
    "7": "",
    "8": "",
    "9": "",
    "10": "",
    "11": "",
    "12": "",
    "13": ""
    }]

directory_trait_types = {
    "0": "background",
    "1": "background-overlay",
    "2": "chinese",
    "3": "snailhead",
    "4": "sprites",
    "5": "speed",
    "6": "top-overlay-1",
    "7": "top-overlay-2",
    "8": "top-overlay-3",
    "9": "top-overlay-4",
    "10": "top-overlay-5"
}


def create_metadata(img_layer, number, metadata_path):
    metadata = {
    "name": "$FAST NFT",
    "description": "The best NFT on Kaspa",
    "image": "",
        "tokenid": number,
    "attributes": []
    }

    for i, asset in enumerate(img_layer):
        trait_type = directory_trait_types[str(i)]
        value = names[i][str(asset)]
        attribute = {
            "traitType": f"{trait_type}",
            "value": f"{value}",
        }
        metadata["attributes"].append(attribute)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    # Set up initial variables
    current_directory = os.getcwd()
    # format output directory with timestamp and date

    current_time = datetime.now().strftime("%H-%M-%S")
    current_date = datetime.now().strftime("%Y-%m-%d") 
    output_directory = os.path.join(current_directory, f"output-{current_time}-{current_date}")
    num_images = 100

    # Load probabilities from JSON file
    probs = load_probabilities("probs.json")
    directories = sorted(probs.keys(), key=lambda x: int(x))  # Sort directories numerically

    # Generate layer choices based on probabilities
    layer_choices = generate_layer_choices(probs, num_images)

    # Prepare ordered probabilities and zip layers
    ordered_probs = [layer_choices[key] for key in directories]
    zipped_layers = list(zip(*ordered_probs))
    # Combine selected layers into images
    images_directory = os.path.join(output_directory, "images")
    metadata_directory = os.path.join(output_directory, "json")
    os.makedirs(output_directory, exist_ok=True)  # Ensure the output directory exists
    os.makedirs(images_directory, exist_ok=True)  # Ensure the output directory exists
    os.makedirs(metadata_directory, exist_ok=True)  # Ensure the output directory exists
    for i, img_layer in enumerate(zipped_layers):
        output_path = os.path.join(images_directory, f"{i}.png")
        metadata_path = os.path.join(metadata_directory, f"{i}")
        print(img_layer)
        create_metadata(img_layer, i, metadata_path)
        combine_layers_to_image(img_layer, metadata_path, directories, current_directory, output_path)

if __name__ == "__main__":
    main()
