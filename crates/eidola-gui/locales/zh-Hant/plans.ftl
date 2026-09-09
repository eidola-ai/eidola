# The shared plans rows (src/plans.rs), Traditional Chinese.

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
