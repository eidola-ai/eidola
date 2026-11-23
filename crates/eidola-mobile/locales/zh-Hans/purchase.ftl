# Purchase (Simplified Chinese). See the English source for what each message is.
purchase-title = 充值额度
purchase-intro = 选择一个方案，或一次性购买额度。订阅额度在其计费周期内有效；一次性购买的额度有效期为一年。未使用且未过期的额度可应要求退款。
purchase-checkout-note = 结账会在你的浏览器中打开，不在本应用内，额度将充入此账户。完成后请回到这里。
purchase-subscribed-note = 此账户已有订阅——请在“账户”中管理。你仍可在这里充值一次性额度。
purchase-loading = 正在加载方案…
purchase-failed = 无法加载方案：{ $reason }
purchase-none = 目前没有可用的方案。
purchase-none-topups = 目前没有可用的一次性充值。
purchase-opened = 结账已在你的浏览器中打开。你回来后会重新读取余额。

# The plan rows: the desktop app's plans.ftl translations.
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
