from typing import List, Set
import random
from ..config.settings import SimulationConfig

class BagSimulator:
    def __init__(self, config: SimulationConfig = None):
        """
        Initialize the BagSimulator with configuration.
        
        Args:
            config: Configuration for the simulation process
        """
        self.config = config or SimulationConfig()
        self.items = self._load_items()
        
    def _load_items(self) -> Set[str]:
        """
        Load the predefined set of items (fruits, vegetables, etc.).
        
        Returns:
            Set[str]: Set of all available items
        """
        return {
            # Fruits (100+ items)
            "apple", "banana", "orange", "grape", "kiwi", "mango", "pear", "peach",
            "plum", "cherry", "strawberry", "blueberry", "raspberry", "blackberry",
            "pineapple", "watermelon", "melon", "cantaloupe", "honeydew", "pomegranate",
            "fig", "date", "apricot", "nectarine", "persimmon", "guava", "papaya",
            "passion fruit", "dragon fruit", "star fruit", "lychee", "rambutan",
            "jackfruit", "durian", "mangosteen", "longan", "loquat", "kumquat",
            "tangerine", "clementine", "mandarin", "grapefruit", "lime", "lemon",
            "coconut", "avocado", "olive", "tomato", "eggplant", "pepper", "mulberry",
            "boysenberry", "gooseberry", "cranberry", "elderberry", "currant", "quince",
            "ugli fruit", "yuzu", "soursop", "custard apple", "breadfruit", "plantain",
            "kiwano", "feijoa", "tamarind", "jujube", "salak", "sapodilla", "marula",
            "miracle fruit", "jabuticaba", "calamansi", "carambola", "cupuacu",
            "acai berry", "goji berry", "bilberry", "cloudberry", "lingonberry",
            "pawpaw", "ackee", "mamey sapote", "cherimoya", "black sapote",
            "white sapote", "cactus pear", "horned melon", "monstera deliciosa",
            "rose apple", "wampee", "yangmei", "safou", "pulasan", "santol",
            "snake fruit", "velvet apple", "wax apple", "white currant", "yumberry",
            
            # Vegetables (80+ items)
            "carrot", "broccoli", "cauliflower", "cabbage", "lettuce", "spinach",
            "kale", "collard greens", "swiss chard", "arugula", "endive", "radicchio",
            "celery", "asparagus", "artichoke", "brussels sprouts", "green beans",
            "snow peas", "sugar snap peas", "zucchini", "squash", "pumpkin",
            "cucumber", "potato", "sweet potato", "yam", "turnip", "rutabaga",
            "parsnip", "beet", "radish", "onion", "garlic", "leek", "shallot",
            "scallion", "mushroom", "corn", "pea", "bean", "lentil", "chickpea",
            "soybean", "edamame", "okra", "rhubarb", "fennel", "ginger", "turmeric",
            "watercress", "bok choy", "napa cabbage", "kohlrabi", "celeriac", "jicama",
            "daikon", "horseradish", "bamboo shoots", "water chestnuts", "lotus root",
            "taro", "cassava", "yuca", "sunchoke", "fiddleheads", "nettles", "purslane",
            "amaranth", "sorrel", "dandelion greens", "mustard greens", "mizuna",
            "romanesco", "salsify", "scorzonera", "sea beans", "samphire", "mache",
            "chayote", "bitter melon", "winter melon", "bottle gourd", "snake gourd",
            "ridge gourd", "sponge gourd", "ivy gourd", "ash gourd", "moringa",

            # Grains & Cereals (30+ items)
            "rice", "quinoa", "barley", "oats", "wheat", "rye", "millet", "sorghum",
            "buckwheat", "amaranth", "teff", "wild rice", "corn", "spelt", "kamut",
            "freekeh", "farro", "bulgur", "triticale", "einkorn", "emmer", "khorasan",
            "job's tears", "black rice", "red rice", "purple rice", "forbidden rice",
            "sticky rice", "jasmine rice", "basmati rice", "arborio rice", "bomba rice",

            # Legumes & Pulses (25+ items)
            "black beans", "kidney beans", "navy beans", "pinto beans", "fava beans",
            "mung beans", "adzuki beans", "lima beans", "black-eyed peas", "split peas",
            "red lentils", "green lentils", "brown lentils", "french lentils",
            "cannellini beans", "great northern beans", "garbanzo beans", "pigeon peas",
            "cranberry beans", "anasazi beans", "calypso beans", "marrow beans",
            "moth beans", "tepary beans", "winged beans", "sword beans", "hyacinth beans",

            # Nuts & Seeds (25+ items)
            "almonds", "walnuts", "pecans", "cashews", "pistachios", "macadamia nuts",
            "brazil nuts", "pine nuts", "hazelnuts", "peanuts", "sunflower seeds",
            "pumpkin seeds", "sesame seeds", "chia seeds", "flax seeds", "hemp seeds",
            "poppy seeds", "lotus seeds", "watermelon seeds", "squash seeds",
            "perilla seeds", "nigella seeds", "caraway seeds", "fennel seeds",
            "cardamom seeds", "mustard seeds", "cumin seeds",

            # Herbs & Spices (40+ items)
            "basil", "oregano", "thyme", "rosemary", "sage", "mint", "cilantro",
            "parsley", "dill", "chives", "tarragon", "marjoram", "bay leaves",
            "cardamom", "cinnamon", "cloves", "nutmeg", "cumin", "coriander",
            "paprika", "saffron", "turmeric", "cayenne", "black pepper",
            "white pepper", "pink pepper", "szechuan pepper", "long pepper",
            "allspice", "anise", "caraway", "juniper", "mace", "galangal",
            "lemongrass", "kaffir lime", "curry leaves", "asafoetida", "sumac",
            "za'atar", "ajowan", "grains of paradise", "mahlab", "annatto",

            # Mushrooms (20+ items)
            "shiitake", "portobello", "oyster", "enoki", "chanterelle", "porcini",
            "morel", "button mushroom", "cremini", "king trumpet", "maitake",
            "beech mushroom", "lion's mane", "reishi", "cordyceps", "turkey tail",
            "chicken of the woods", "hen of the woods", "black trumpet", "wood ear",
            "nameko", "shimeji", "matsutake",

            # Seaweed & Sea Vegetables (15+ items)
            "nori", "wakame", "kombu", "dulse", "arame", "hijiki", "sea lettuce",
            "irish moss", "kelp", "agar", "sea grapes", "bladderwrack", "sea palm",
            "ogonori", "mozuku", "tosaka",

            # Dairy & Eggs (20+ items)
            "milk", "cream", "butter", "cheese", "yogurt", "sour cream", "cottage cheese",
            "ricotta", "mozzarella", "cheddar", "parmesan", "gouda", "brie", "camembert",
            "feta", "blue cheese", "eggs", "quail eggs", "duck eggs", "goose eggs",
            "cream cheese", "mascarpone", "ghee", "kefir",

            # Meat & Poultry (20+ items)
            "beef", "pork", "lamb", "chicken", "turkey", "duck", "goose", "quail",
            "venison", "rabbit", "bison", "elk", "boar", "veal", "goat", "mutton",
            "pheasant", "guinea fowl", "partridge", "squab", "ostrich", "emu",

            # Seafood (25+ items)
            "salmon", "tuna", "cod", "halibut", "trout", "sardines", "mackerel",
            "shrimp", "crab", "lobster", "oysters", "mussels", "clams", "scallops",
            "squid", "octopus", "sea bass", "snapper", "grouper", "swordfish",
            "tilapia", "catfish", "anchovy", "herring", "barramundi", "crayfish",
            "abalone", "sea urchin", "caviar"
        }

    def generate_random_bag(self) -> Set[str]:
        """
        Generate a random bag of items.
        
        Returns:
            Set[str]: A set of randomly selected items
        """
        size = random.randint(self.config.min_bag_size, self.config.max_bag_size)
        return set(random.sample(list(self.items), size))

    def generate_bags(self) -> List[Set[str]]:
        """
        Generate multiple random bags.
        
        Returns:
            List[Set[str]]: List of randomly generated bags
        """
        return [self.generate_random_bag() for _ in range(self.config.num_bags)]

    def get_bag_statistics(self, bags: List[Set[str]]) -> dict:
        """
        Calculate statistics for a list of bags.
        
        Args:
            bags: List of bags to analyze
            
        Returns:
            dict: Dictionary containing bag statistics
        """
        sizes = [len(bag) for bag in bags]
        return {
            "num_bags": len(bags),
            "min_size": min(sizes),
            "max_size": max(sizes),
            "avg_size": sum(sizes) / len(sizes),
            "total_unique_items": len(set().union(*bags)),
            "avg_unique_items_per_bag": sum(len(bag) for bag in bags) / len(bags)
        } 