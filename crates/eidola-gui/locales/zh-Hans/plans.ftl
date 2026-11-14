# The shared plans rows (src/plans.rs), Simplified Chinese.

plans-list = 可选方案
plans-opening-checkout = 正在打开结账页面…

plans-free = 免费
plans-price-cadence =
    { $count ->
        [1]
            { $interval ->
                [day] { $amount }/天
                [week] { $amount }/周
                [month] { $amount }/月
                [year] { $amount }/年
               *[other] { $amount }/{ $interval }
            }
       *[other]
            { $interval ->
                [day] 每 { $count } 天 { $amount }
                [week] 每 { $count } 周 { $amount }
                [month] 每 { $count } 个月 { $amount }
                [year] 每 { $count } 年 { $amount }
               *[other] 每 { $count } { $interval } { $amount }
            }
    }

plans-credits-one-time = { $credits } 点额度，自购买之日起一年后过期
plans-credits-recurring = { $credits } 点额度，在每个计费周期结束时过期
plans-credits-described = { $line } —— { $description }
