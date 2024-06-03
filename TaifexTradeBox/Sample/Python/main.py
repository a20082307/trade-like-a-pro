
from TTBHelp import *

class TTBProcess(TTBModule):
    def SHOWQUOTEDATA(self,obj):
        print("Symbol:{}, BidPs:{}, BidPv:{}, AskPs:{}, AskPv:{}, T:{}, P:{}, V:{}, Volume:{}".format(obj['Symbol'],obj['BidPs'],obj['BidPv'],obj['AskPs'],obj['AskPv'],obj['TickTime'],obj['Price'],obj['Qty'],obj['Volume']))  
        # if(float(obj['Price'].replace(',','')) >=14000):
        #     putOrd = {
        #         "Symbol1":"TXFG3",
        #         "Price":"11000",
        #         "TimeInForce":"1",
        #         "Side1":"1",  
        #         "OrderType":"2",
        #         "OrderQty":"1",
        #         "DayTrade":"0",
        #         "Symbol2":"",
        #         "Side2":"",
        #         "PositionEffect": "4"
        #     }
        # msgDic = super().NEWORDER(putOrd)

if __name__ == "__main__":
    ttbModule = TTBProcess('http://localhost:8080',51141)
    ttbModule.QUOTEDATA('CDFF4')
    
    # putOrd = {
    #     "Symbol1":"TXFA4",
    #     "Price":"16001",
    #     "TimeInForce":"1",
    #     "Side1":"1",
    #     "OrderType":"2",
    #     "OrderQty":"5",
    #     "DayTrade":"0",
    #     "Symbol2":"",
    #     "Side2":"",
    #     "PositionEffect": "",
    # }
    # msgDic = ttbModule.NEWORDER(putOrd)
    # print(msgDic)

    # # cancelOrder
    # canlOrd = {
    #     "OrdNo":"fC56001a"
    # }
    # msgDic = ttbModule.CANCELORDER(canlOrd)
    # print(msgDic)

    #changePrice
    # changePrice = {
    #     "OrdNo":"fC56001a",
    #     "Price":"16002",
    # }
    # msgDic = ttbModule.REPLACEPRICE(changePrice)
    # print(msgDic)

    # # changeQty
    # changeQty = {
    #     "OrdNo":"fC56001a",
    #     "UdpQty":"4"
    # }
    # msgDic = ttbModule.REPLACEQTY(changeQty)
    # print(msgDic)

    # msgDic = ttbModule.QUERYMARGIN()
    # print(msgDic)

    # msgDic = ttbModule.QUERYRESTOREFILLREPORT()
    # print(msgDic)

    # msgDic = ttbModule.QUERYRESTOREREPORT()
    # print(msgDic)

