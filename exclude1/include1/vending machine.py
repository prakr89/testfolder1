def vending_machine():
    a = {'item': 'choc', 'price': 1.5, 'stock': 2}
    b = {'item': 'pop', 'price': 1.75, 'stock': 1}
    c = {'item': 'chips', 'price': 2.0, 'stock': 3}
    d = {'item': 'gum', 'price': 0.50, 'stock': 1}
    e = {'item': 'mints', 'price': 0.75, 'stock': 3}
    items=[a,b,c,d,e]
    now_cash=0
    print("Welcome to vending machine")
    def update_items(items):
        for item in items:
            if item.get("stock")==0:
                items.remeove(item)
        for item in items:
            print(item.get('item'), item.get('price'))

    continueToBuy = True
    while continueToBuy ==True:
        update_items(items)
        selected=input("enter item name from from above list")
        for item in items:
            if selected == item.get('item'):
                selected = item
                selected_price=selected.get("price")
                while now_cash<selected_price:
                    now_cash=now_cash+float(input("please add "+str(selected_price-now_cash)+"coin/rupees value to match price"))
                print('you got ' + str(selected.get('item')))
                selected['stock'] -= 1
                now_cash -= selected_price
                print("cash remaining is " +now_cash )
                a = input('buy something else? (y/n): ')
                if a == 'n':
                    continueToBuy = False
                    #so start refunding
                    if now_cash != 0:
                        print(str(now_cash) + ' refunded')
                        now_cash = 0
                        print('thank you, have a nice day!\n')
                        break
                    else:
                        print('thank you, have a nice day!\n')
                        break
                else:
                    continue



vending_machine()
