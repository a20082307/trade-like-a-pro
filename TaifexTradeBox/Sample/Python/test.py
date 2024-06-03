import TTBHelp

class ATBModule(TTBHelp):
    def __init__(self, host, zmpProt):
        print('tst')
        super().__init__(host, zmpProt)

    def QUOTEDATA(self, symbols):
        super().REGISTERQUOTEDATA(symbols)

    def SHOWQUOTEDATA(self, obj):
        print("test")
        print(obj)

if __name__ == "__main__":
    atb = ATBModule('http://localhost:8080', 51141)
    atb.QUOTEDATA('TXFF4')