# The conversation window (src/space_view/mod.rs), Simplified Chinese.

space-error-archived = 此对话已归档，无法接收新的回复。
space-error-not-joined = 这是你的助理之间开启的对话。你可以阅读全部内容；若要在这里修改或重新提问，请先发言参与。

space-footnote-delegation-concluded = 已告一段落
space-footnote-delegation-concluded-truncated = 已告一段落，但结束在一条因长度上限而中断的回复上
space-footnote-delegation-paused = 已达到回复上限（{ $depth }/{ $limit }）
space-footnote-delegation-paused-truncated = 已达到回复上限（{ $depth }/{ $limit }），且停在一条因长度上限而中断的回复上
space-footnote-delegation-budget = 已用完全部 { $limit } 个回合
space-footnote-delegation-budget-truncated = 已用完全部 { $limit } 个回合，且停在一条因长度上限而中断的回复上
space-footnote-delegation-failed = 已中止：{ $reason ->
        [upstream] 无法连接到模型
        [funding] 无法为该回合付费
        [configuration] 其设置存在问题
       *[other] 有一个回合未能完成
    }

space-regenerating = 正在重新生成…
space-error-model-unavailable = { $model } 所在的后端已不再提供该模型，也不会由其他模型代为作答。没有发送任何请求，也没有产生任何费用——请改选一个模型后重试。
space-error-hold-exceeds-denomination = 这条回复的费用可能超出单个请求允许预留的额度，因此该请求没有发送，也没有为它产生任何费用。充值无法解决这个问题——请缩短对话或改用其他模型。
space-regenerating-elsewhere = 这条回复正在重新生成。
space-answer-cut-off = 这条回复在长度上限处停下。
space-continue-answer = 继续
space-continue-answer-label = 从停下的地方继续这条回复
space-continuing = 正在继续…
space-error-issuer-key-off-calendar = Eidola 服务器提供的支付密钥不符合公开的密钥日程，因此本应用没有使用它。没有发送任何请求，也没有产生任何费用；你已持有的额度仍可正常使用。
space-error-issuer-key-second-for-epoch = Eidola 服务器本周提供的支付密钥与本应用已有的不同。诚实的服务器每周只使用一个密钥，因此在下周的密钥生效之前，本应用不会用它领取新的额度。没有发送任何请求，也没有产生任何费用；你已持有的额度仍可正常使用。
space-error-issuer-key-reused = Eidola 服务器为两个不同的周提供了同一个支付密钥，诚实的服务器绝不会这样做，因此本应用没有使用它。没有发送任何请求，也没有产生任何费用；你已持有的额度仍可正常使用。
space-error-uncached-retry-unfunded = Eidola 引擎已不再保留此对话之前的上下文，而本应用持有的额度不足以完整地重新发送。没有再从你的账户转换任何额度。请重试，以新请求的方式发送。
