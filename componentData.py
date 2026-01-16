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
    "TFT_Item_FryingPan",
}

itemToComponent = {
    # --- PHYSICAL / AD CARRY ---
    "TFT_Item_Deathblade": ["TFT_Item_BFSword", "TFT_Item_BFSword"],
    "TFT_Item_InfinityEdge": ["TFT_Item_BFSword", "TFT_Item_SparringGloves"],
    "TFT_Item_MadredsBloodrazor": ["TFT_Item_BFSword", "TFT_Item_RecurveBow"], # Giant Slayer
    "TFT_Item_SteraksGage": ["TFT_Item_BFSword", "TFT_Item_GiantsBelt"],
    "TFT_Item_GuardianAngel": ["TFT_Item_BFSword", "TFT_Item_ChainVest"], # Edge of Night
    "TFT_Item_Bloodthirster": ["TFT_Item_BFSword", "TFT_Item_NegatronCloak"],
    "TFT_Item_HextechGunblade": ["TFT_Item_BFSword", "TFT_Item_NeedlesslyLargeRod"],
    "TFT_Item_SpearOfShojin": ["TFT_Item_BFSword", "TFT_Item_TearOfTheGoddess"],

    # --- ATTACK SPEED ---
    "TFT_Item_RedBuff": ["TFT_Item_RecurveBow", "TFT_Item_RecurveBow"],
    "TFT_Item_LastWhisper": ["TFT_Item_RecurveBow", "TFT_Item_SparringGloves"],
    "TFT_Item_NashorsTooth": ["TFT_Item_RecurveBow", "TFT_Item_GiantsBelt"],
    "TFT_Item_TitansResolve": ["TFT_Item_RecurveBow", "TFT_Item_ChainVest"],
    "TFT_Item_RunaansHurricane": ["TFT_Item_RecurveBow", "TFT_Item_NegatronCloak"],
    "TFT_Item_GuinsoosRageblade": ["TFT_Item_RecurveBow", "TFT_Item_NeedlesslyLargeRod"],
    "TFT_Item_StatikkShiv": ["TFT_Item_RecurveBow", "TFT_Item_TearOfTheGoddess"],

    # --- MAGIC / AP ---
    "TFT_Item_JeweledGauntlet": ["TFT_Item_NeedlesslyLargeRod", "TFT_Item_SparringGloves"],
    "TFT_Item_Morellonomicon": ["TFT_Item_NeedlesslyLargeRod", "TFT_Item_GiantsBelt"],
    "TFT_Item_SpectralGauntlet": ["TFT_Item_NeedlesslyLargeRod", "TFT_Item_ChainVest"], # Crownguard
    "TFT_Item_IonicSpark": ["TFT_Item_NeedlesslyLargeRod", "TFT_Item_NegatronCloak"],
    "TFT_Item_RabadonsDeathcap": ["TFT_Item_NeedlesslyLargeRod", "TFT_Item_NeedlesslyLargeRod"],
    "TFT_Item_ArchangelsStaff": ["TFT_Item_NeedlesslyLargeRod", "TFT_Item_TearOfTheGoddess"],

    # --- MANA / UTILITY ---
    "TFT_Item_UnstableConcoction": ["TFT_Item_TearOfTheGoddess", "TFT_Item_SparringGloves"], # Hand of Justice
    "TFT_Item_Redemption": ["TFT_Item_TearOfTheGoddess", "TFT_Item_GiantsBelt"],
    "TFT_Item_ProtectorsVow": ["TFT_Item_TearOfTheGoddess", "TFT_Item_ChainVest"],
    "TFT_Item_AdaptiveHelm": ["TFT_Item_TearOfTheGoddess", "TFT_Item_NegatronCloak"],
    "TFT_Item_BlueBuff": ["TFT_Item_TearOfTheGoddess", "TFT_Item_TearOfTheGoddess"],

    # --- TANK / DEFENSE ---
    "TFT_Item_NightHarvester": ["TFT_Item_ChainVest", "TFT_Item_SparringGloves"], # Steadfast Heart
    "TFT_Item_SunfireCape": ["TFT_Item_ChainVest", "TFT_Item_GiantsBelt"],
    "TFT_Item_BrambleVest": ["TFT_Item_ChainVest", "TFT_Item_ChainVest"],
    "TFT_Item_GargoyleStoneplate": ["TFT_Item_ChainVest", "TFT_Item_NegatronCloak"],
    "TFT_Item_Evenshroud": ["TFT_Item_NegatronCloak", "TFT_Item_GiantsBelt"],
    "TFT_Item_DragonsClaw": ["TFT_Item_NegatronCloak", "TFT_Item_NegatronCloak"],
    "TFT_Item_Quicksilver": ["TFT_Item_NegatronCloak", "TFT_Item_SparringGloves"],
    "TFT_Item_PowerGauntlet": ["TFT_Item_GiantsBelt", "TFT_Item_SparringGloves"], # Guardbreaker
    "TFT_Item_WarmogsArmor": ["TFT_Item_GiantsBelt", "TFT_Item_GiantsBelt"],
    "TFT_Item_ThiefsGloves": ["TFT_Item_SparringGloves", "TFT_Item_SparringGloves"],

    # --- SPATULA (ORIGIN EMBLEMS) ---
    "TFT_Item_IoniaEmblemItem": ["TFT_Item_Spatula", "TFT_Item_BFSword"],
    "TFT_Item_ChallengerEmblemItem": ["TFT_Item_Spatula", "TFT_Item_RecurveBow"],
    "TFT_Item_ShurimaEmblemItem": ["TFT_Item_Spatula", "TFT_Item_NeedlesslyLargeRod"],
    "TFT_Item_SorcererEmblemItem": ["TFT_Item_Spatula", "TFT_Item_TearOfTheGoddess"],
    "TFT16_Item_DemaciaEmblemItem": ["TFT_Item_Spatula", "TFT_Item_ChainVest"],
    "TFT16_Item_YordleEmblemItem": ["TFT_Item_Spatula", "TFT_Item_NegatronCloak"],
    "TFT_Item_FreljordEmblemItem": ["TFT_Item_Spatula", "TFT_Item_GiantsBelt"],
    "TFT16_Item_IxtalEmblemItem": ["TFT_Item_Spatula", "TFT_Item_SparringGloves"],
    "TFT_Item_ForceOfNature": ["TFT_Item_Spatula", "TFT_Item_Spatula"], # Tactician's Crown

    # --- FRYING PAN (CLASS EMBLEMS) ---
    "TFT16_Item_SlayerEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_BFSword"],
    "TFT16_Item_GunslingerEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_RecurveBow"],
    "TFT16_Item_ZaunEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_NeedlesslyLargeRod"], # Likely mapping for Magic+Pan in this set
    "TFT_Item_InvokerEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_TearOfTheGoddess"],
    "TFT_Item_DefenderEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_ChainVest"], # Bastion
    "TFT_Item_TargonEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_NegatronCloak"],
    "TFT16_Item_BrawlerEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_GiantsBelt"], # Bruiser
    "TFT_Item_VanquisherEmblemItem": ["TFT_Item_FryingPan", "TFT_Item_SparringGloves"],
    "TFT_Item_TacticiansShield": ["TFT_Item_FryingPan", "TFT_Item_FryingPan"], # Pan + Pan
    "TFT_Item_TacticiansCape": ["TFT_Item_Spatula", "TFT_Item_FryingPan"],
} 