import pickle 
# a standard python library
# pickle (AKA object serialization): writing a binary representation of an object to a file
# unpickle (AKA object de-serialization): read a binary representation of an object from a file
# to load a python object in to memory


# for the project... imagine this is a MyRandomForestClassifier
# for PA6... imagine this is a MyDecisionTreeClassifier
# both of which were just trained with fit()
coins = ["Dogecoin", "XRP", "NEM", "USDCoin", "Cosmos", "Solana", "Tron", "Uniswap", "ChainLink", "Polkadot", "WrappedBitcoin", "Litecoin", "Iota", "Tether", "Aave", "Ethereum", "EOS", "Cardano", "CryptocomCoin", "BinanceCoin", "Stellar", "Monero", "Bitcoin"]

header = ["att0", "att1", "att2", "att3"]

'''
interview_tree = \
["Attribute", "level", 
    ["Value", "Senior", 
        ["Attribute", "tweets", 
            ["Value", "yes", 
                ["Leaf", "True", 2, 5]

            ],

            ["Value", "no", 
                ["Leaf", "False", 3, 5]

            ]

        ]

    ],

    ["Value", "Mid", 
        ["Leaf", "True", 4, 14]

    ],

    ["Value", "Junior", 
        ["Attribute", "phd", 
            ["Value", "yes", 
                ["Leaf", "False", 2, 5]

            ],

            ["Value", "no", 
                ["Leaf", "True", 3, 5]

            ]

        ]

    ]

]

'''
tree = [[['Attribute', 'att1', ['Value', 'Bearish', ['Leaf', 'Gains', 388, 515]], ['Value', 'Bullish', ['Leaf', 'Loses', 263, 351]]], 0.76875], [['Attribute', 'att1', ['Value', 'Bearish', ['Leaf', 'Gains', 388, 515]], ['Value', 'Bullish', ['Leaf', 'Loses', 263, 351]]], 0.759375], [['Attribute', 'att1', ['Value', 'Bearish', ['Leaf', 'Gains', 388, 515]], ['Value', 'Bullish', ['Leaf', 'Loses', 263, 351]]], 0.765625], [['Attribute', 'att1', ['Value', 'Bearish', ['Leaf', 'Gains', 388, 515]], ['Value', 'Bullish', ['Leaf', 'Loses', 263, 351]]], 0.75625], [['Attribute', 'att1', ['Value', 'Bearish', ['Attribute', 'att3', ['Value', 1, ['Leaf', 'Gains', 109, 515]], ['Value', 2, ['Leaf', 'Gains', 87, 515]], ['Value', 0, ['Leaf', 'Gains', 108, 515]], ['Value', 3, ['Leaf', 'Gains', 84, 515]]]], ['Value', 'Bullish', ['Attribute', 'att3', ['Value', 1, ['Leaf', 'Loses', 67, 351]], ['Value', 2, ['Leaf', 'Loses', 67, 351]], ['Value', 0, ['Leaf', 'Loses', 63, 351]], ['Value', 3, ['Leaf', 'Loses', 66, 351]]]]], 0.753125], [['Attribute', 'att1', ['Value', 'Bearish', ['Leaf', 'Gains', 388, 515]], ['Value', 'Bullish', ['Leaf', 'Loses', 263, 351]]], 0.7875], [['Attribute', 'att1', ['Value', 'Bearish', ['Leaf', 'Gains', 388, 515]], ['Value', 'Bullish', ['Leaf', 'Loses', 263, 351]]], 0.740625]]

# pickle (save to file) header and interview tree as one object
packaged_object = [coins, header, tree]

outfile = open("tree.p", "wb")
pickle.dump(packaged_object, outfile)
outfile.close()
