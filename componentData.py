components = {
    "TFT_Item_BFSword",
    "TFT_Item_RecurveBow",
    "TFT_Item_NeedlesslyLargeRod",
    "TFT_Item_TearOfTheGoddess",
    "TFT_Item_ChainVest",
    "TFT_Item_NegatronCloak",
    "TFT_Item_GiantsBelt",
    "TFT_Item_SparringGloves",
    "TFT_Item_Spatula",
    "TFT_Item_FryingPan",  # New core component for Class Emblems
}

itemToComponent = {
    # --- PHYSICAL / AD CARRY ---
    "TFT_Item_Deathblade": ["TFT_Item_BFSword", "TFT_Item_BFSword"],
    "TFT_Item_InfinityEdge": ["TFT_Item_BFSword", "TFT_Item_SparringGloves"],
    "TFT_Item_GiantSlayer": ["TFT_Item_BFSword", "TFT_Item_RecurveBow"],
    "TFT_Item_SteraksGage": ["TFT_Item_BFSword", "TFT_Item_GiantsBelt"],
    "TFT_Item_EdgeOfNight": ["TFT_Item_BFSword", "TFT_Item_ChainVest"],
    "TFT_Item_Bloodthirster": ["TFT_Item_BFSword", "TFT_Item_NegatronCloak"],
    
    # --- ATTACK SPEED / UTILITY ---
    "TFT_Item_RapidFireCannon": ["TFT_Item_RecurveBow", "TFT_Item_RecurveBow"],  # Red Buff
    "TFT_Item_GuinsoosRageblade": ["TFT_Item_RecurveBow", "TFT_Item_NeedlesslyLargeRod"],
    "TFT_Item_LastWhisper": ["TFT_Item_RecurveBow", "TFT_Item_SparringGloves"],
    "TFT_Item_TitanResolve": ["TFT_Item_RecurveBow", "TFT_Item_ChainVest"],
    "TFT_Item_NashorsTooth": ["TFT_Item_RecurveBow", "TFT_Item_GiantsBelt"],
    "TFT_Item_KrakensFury": ["TFT_Item_RecurveBow", "TFT_Item_NegatronCloak"],  # Set 16: Replaced Runaan's
    
    # --- MAGIC / AP CARRY ---
    "TFT_Item_RabadonsDeathcap": ["TFT_Item_NeedlesslyLargeRod", "TFT_Item_NeedlesslyLargeRod"],
    "TFT_Item_JeweledGauntlet": ["TFT_Item_NeedlesslyLargeRod", "TFT_Item_SparringGloves"],
    "TFT_Item_ArchangelsStaff": ["TFT_Item_NeedlesslyLargeRod", "TFT_Item_TearOfTheGoddess"],
    "TFT_Item_Crownguard": ["TFT_Item_NeedlesslyLargeRod", "TFT_Item_ChainVest"],
    "TFT_Item_IonicSpark": ["TFT_Item_NeedlesslyLargeRod", "TFT_Item_NegatronCloak"],
    "TFT_Item_Morellonomicon": ["TFT_Item_NeedlesslyLargeRod", "TFT_Item_GiantsBelt"],
    "TFT_Item_HextechGunblade": ["TFT_Item_NeedlesslyLargeRod", "TFT_Item_BFSword"],
    "TFT_Item_VoidStaff": ["TFT_Item_TearOfTheGoddess", "TFT_Item_RecurveBow"],  # Set 16: Replaced Statikk Shiv

    # --- MANA / SUPPORT ---
    "TFT_Item_BlueBuff": ["TFT_Item_TearOfTheGoddess", "TFT_Item_TearOfTheGoddess"],
    "TFT_Item_SpearOfShojin": ["TFT_Item_TearOfTheGoddess", "TFT_Item_BFSword"],
    "TFT_Item_HandOfJustice": ["TFT_Item_TearOfTheGoddess", "TFT_Item_SparringGloves"],
    "TFT_Item_ProtectorsVow": ["TFT_Item_TearOfTheGoddess", "TFT_Item_ChainVest"],
    "TFT_Item_AdaptiveHelm": ["TFT_Item_TearOfTheGoddess", "TFT_Item_NegatronCloak"],
    "TFT_Item_SpiritVisage": ["TFT_Item_TearOfTheGoddess", "TFT_Item_GiantsBelt"],  # Set 16: Replaced Redemption

    # --- TANK / DEFENSE ---
    "TFT_Item_WarmogsArmor": ["TFT_Item_GiantsBelt", "TFT_Item_GiantsBelt"],
    "TFT_Item_BrambleVest": ["TFT_Item_ChainVest", "TFT_Item_ChainVest"],
    "TFT_Item_DragonsClaw": ["TFT_Item_NegatronCloak", "TFT_Item_NegatronCloak"],
    "TFT_Item_GargoyleStoneplate": ["TFT_Item_ChainVest", "TFT_Item_NegatronCloak"],
    "TFT_Item_SunfireCape": ["TFT_Item_ChainVest", "TFT_Item_GiantsBelt"],
    "TFT_Item_SteadfastHeart": ["TFT_Item_ChainVest", "TFT_Item_SparringGloves"],
    "TFT_Item_Evenshroud": ["TFT_Item_NegatronCloak", "TFT_Item_GiantsBelt"],
    "TFT_Item_Quicksilver": ["TFT_Item_NegatronCloak", "TFT_Item_SparringGloves"],
    "TFT_Item_StrikersFlail": ["TFT_Item_GiantsBelt", "TFT_Item_SparringGloves"],  # Set 16: Replaced Guardbreaker
    "TFT_Item_ThiefsGloves": ["TFT_Item_SparringGloves", "TFT_Item_SparringGloves"],

    # --- SPATULA (ORIGIN EMBLEMS) ---
    "TFT_Item_NoxusEmblemItem": ["TFT_Item_Spatula", "TFT_Item_BFSword"],
    "TFT_Item_VoidEmblemItem": ["TFT_Item_Spatula", "TFT_Item_RecurveBow"],
    "TFT_Item_IoniaEmblemItem": ["TFT_Item_Spatula", "TFT_Item_NeedlesslyLargeRod"],
    "TFT_Item_PiltoverEmblemItem": ["TFT_Item_Spatula", "TFT_Item_TearOfTheGoddess"],
    "TFT_Item_DemaciaEmblemItem": ["TFT_Item_Spatula", "TFT_Item_ChainVest"],
    "TFT_Item_YordleEmblemItem": ["TFT_Item_Spatula", "TFT_Item_NegatronCloak"],
    "TFT_Item_FreljordEmblemItem": ["TFT_Item_Spatula", "TFT_Item_GiantsBelt"],
    "TFT_Item_IxtalEmblemItem": ["TFT_Item_Spatula", "TFT_Item_SparringGloves"],
    "TFT_Item_ForceOfNature": ["TFT_Item_Spatula", "TFT_Item_Spatula"],  # Tactician's Crown

    # --- FRYING PAN (CLASS EMBLEMS) ---
    "TFT_Item_SlayerEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_BFSword"],
    "TFT_Item_QuickstrikerEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_RecurveBow"],
    "TFT_Item_ArcanistEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_NeedlesslyLargeRod"],
    "TFT_Item_InvokerEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_TearOfTheGoddess"],
    "TFT_Item_DefenderEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_ChainVest"],
    "TFT_Item_JuggernautEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_NegatronCloak"],
    "TFT_Item_BruiserEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_GiantsBelt"],
    "TFT_Item_VanquisherEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_SparringGloves"],
    "TFT_Item_TacticiansCape": ["TFT_Item_FryingPan", "TFT_Item_FryingPan"],
    "TFT_Item_TacticiansShield": ["TFT_Item_FryingPan", "TFT_Item_Spatula"],
}