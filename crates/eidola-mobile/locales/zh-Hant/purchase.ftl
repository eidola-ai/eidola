# Purchase (Traditional Chinese). See the English source for what each message is.
purchase-title = 儲值額度
purchase-intro = 選擇一個方案，或一次性購買額度。訂閱額度在其計費週期內有效；一次性購買的額度有效期為一年。未使用且未過期的額度可應要求退款。
purchase-checkout-note = 結帳會在你的瀏覽器中開啟，不在本應用程式內，額度將存入此帳戶。完成後請回到這裡。
purchase-subscribed-note = 此帳戶已有訂閱——請在「帳戶」中管理。你仍可在這裡儲值一次性額度。
purchase-loading = 正在載入方案…
purchase-failed = 無法載入方案：{ $reason }
purchase-none = 目前沒有可用的方案。
purchase-none-topups = 目前沒有可用的一次性儲值。
purchase-opened = 結帳已在你的瀏覽器中開啟。你回來後會重新讀取餘額。

# The plan rows: the desktop app's plans.ftl translations.
plans-list = 可選方案
plans-opening-checkout = 正在開啟結帳頁面…
plans-free = 免費
plans-price-cadence =
    { $count ->
        [1]
            { $interval ->
                [day] { $amount }/天
                [week] { $amount }/週
                [month] { $amount }/月
                [year] { $amount }/年
               *[other] { $amount }/{ $interval }
            }
       *[other]
            { $interval ->
                [day] 每 { $count } 天 { $amount }
                [week] 每 { $count } 週 { $amount }
                [month] 每 { $count } 個月 { $amount }
                [year] 每 { $count } 年 { $amount }
               *[other] 每 { $count } { $interval } { $amount }
            }
    }
plans-credits-one-time = { $credits } 點額度，自購買之日起一年後過期
plans-credits-recurring = { $credits } 點額度，在每個計費週期結束時過期
plans-credits-described = { $line } —— { $description }
