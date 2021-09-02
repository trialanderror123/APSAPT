# For Sentiment Analysis
LABELS = ['Pro Trump', 'Pro Biden', 'Neutral']
TRAIN_DATA_PATH = "./data/train_unbalanced_final.csv"
PREDICTION_DATA_PATH = "./data/raw_data.csv"
TEST_SIZE = 0.2
MAX_LENGTH = 100
MODEL_PATH = "./bert_model"
BATCH_SIZE = 8
NO_DECAY = ['bias', 'gamma', 'beta']
WEIGHT_DECAY_RATE = 0.01
LR = 3e-5
EPOCHS = 5

# For Crawling (Scraping) Tweets
NUM_TWEETS_PER_TAG = 5000
START_DATE="2020-09-01"
END_DATE="2020-12-31"
KEYWORDS = [
                    # Mentions of Trump: 7
                    "#Trump", "#trump", "#Trump2020", "#DonaldTrump", "DonaldJTrump", "Donald Trump", "Trump"

                    # Pro-Trump: 8
                    '#VoteTrump', "VoteRed", "#MAGA", "#PresidentTrump",  '#MakeAmericaGreatAgain', '#TeamTrump',  '#DrainTheSwamp',  "#MyPresident",
                    
                    # Anti-Trump: 7
                    "#VoteTrumpOut", "#DumpTrump", '#TrumpIsPathetic', '#TrumpCorruption', '#VoteHimOut', '#YoureFiredTrump', '#TrumpHasToGo',
                    
                    # Mentions of Biden: 6
                    "#Biden", "#biden", "#Biden2020", "Joe Biden", "#JoeBiden", "Biden",
                    
                    # Pro-Biden: 6
                    "#VoteBiden", "VoteBlue", "#VoteBlueToSaveAmerica", "#BlueWave2020", '#TeamBiden', '#JoeMentum', 
                    
                    # Anti-Biden: 7
                    "Sleepy Joe", "#SleepyJoe", "HidenBiden", "#CreepyJoeBiden", "#NeverBiden", "#BidenUkraineScandal", '#HunterBiden',
                    
                    # Miscellaneous: 1
                    "#USElections"
]
COUNTRIES_DICT = {
            "Iran":"Tehran", "Israel":"Jerusalem", 
            "Saudi Arabia":"Riyadh", "China":"Hong Kong",
            "Ukraine":"Kyiv", "Russia":"Moscow",
            "UK":"London", "India":"New Delhi", 
            "Mexico":"Mexico City", "Canada":"Ottawa", 
            "Brazil":"Brasilia", "South Korea":"Seoul",
            "Philippines":"Manila", "Kenya":"Nairobi",
            "Nigeria":"Abuja","Germany":"Berlin",
            "Taiwan":"Taipei","France":"Paris",
            "Afghanistan":"Kabul", "Indonesia":"Jakarta",
            "Japan":"Tokyo", "Australia":"Canberra",
            "Singapore":"Singapore"
        }