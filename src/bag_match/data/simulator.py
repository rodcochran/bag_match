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
            # Fruits (71 items)
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
            
            # Vegetables (65 items)
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

            # Grains & Cereals (18 items)
            "rice", "quinoa", "barley", "oats", "wheat", "rye", "millet", "sorghum",
            "buckwheat", "amaranth", "teff", "wild rice", "corn", "spelt", "kamut",
            "freekeh", "farro", "bulgur",

            # Legumes & Pulses (14 items)
            "black beans", "kidney beans", "navy beans", "pinto beans", "fava beans",
            "mung beans", "adzuki beans", "lima beans", "black-eyed peas", "split peas",
            "red lentils", "green lentils", "brown lentils", "french lentils",

            # Nuts & Seeds (17 items)
            "almonds", "walnuts", "pecans", "cashews", "pistachios", "macadamia nuts",
            "brazil nuts", "pine nuts", "hazelnuts", "peanuts", "sunflower seeds",
            "pumpkin seeds", "sesame seeds", "chia seeds", "flax seeds", "hemp seeds",
            "poppy seeds",

            # Herbs & Spices (24 items)
            "basil", "oregano", "thyme", "rosemary", "sage", "mint", "cilantro",
            "parsley", "dill", "chives", "tarragon", "marjoram", "bay leaves",
            "cardamom", "cinnamon", "cloves", "nutmeg", "cumin", "coriander",
            "paprika", "saffron", "turmeric", "cayenne", "black pepper",

            # Mushrooms (12 items)
            "shiitake", "portobello", "oyster", "enoki", "chanterelle", "porcini",
            "morel", "button mushroom", "cremini", "king trumpet", "maitake",
            "beech mushroom",

            # Seaweed & Sea Vegetables (9 items)
            "nori", "wakame", "kombu", "dulse", "arame", "hijiki", "sea lettuce",
            "irish moss", "kelp"
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