# The conversation window (src/space_view/mod.rs), Traditional Chinese.

space-error-archived = 此對話已封存，無法接收新的回覆。
space-error-not-joined = 這是你的助理之間開啟的對話。你可以閱讀全部內容；若要在這裡修改或重新提問，請先發言參與。

space-footnote-delegation-concluded = 已告一段落
space-footnote-delegation-concluded-truncated = 已告一段落，但結束在一則因長度上限而中斷的回覆上
space-footnote-delegation-paused = 已達到回覆上限（{ $depth }/{ $limit }）
space-footnote-delegation-paused-truncated = 已達到回覆上限（{ $depth }/{ $limit }），且停在一則因長度上限而中斷的回覆上
space-footnote-delegation-budget = 已用完全部 { $limit } 個回合
space-footnote-delegation-budget-truncated = 已用完全部 { $limit } 個回合，且停在一則因長度上限而中斷的回覆上
space-footnote-delegation-failed = 已中止：{ $reason ->
        [upstream] 無法連線到模型
        [funding] 無法為該回合付費
        [configuration] 其設定存在問題
       *[other] 有一個回合未能完成
    }

space-regenerating = 正在重新產生…
space-error-model-unavailable = { $model } 所在的後端已不再提供該模型，也不會由其他模型代為作答。沒有發送任何請求，也沒有產生任何費用——請改選一個模型後重試。
space-error-hold-exceeds-denomination = 這則回覆的費用可能超出單一請求允許預留的額度，因此該請求沒有發送，也沒有為它產生任何費用。儲值無法解決這個問題——請縮短對話或改用其他模型。
space-regenerating-elsewhere = 這則回覆正在重新產生。
space-answer-cut-off = 這則回覆在長度上限處停下。
space-continue-answer = 繼續
space-continue-answer-label = 從停下的地方繼續這則回覆
space-continuing = 正在繼續…
space-error-issuer-key-off-calendar = Eidola 伺服器提供的付款金鑰不符合公開的金鑰時程，因此本應用程式沒有使用它。沒有傳送任何請求，也沒有產生任何費用；你已持有的額度仍可正常使用。
space-error-issuer-key-second-for-epoch = Eidola 伺服器本週提供的付款金鑰與本應用程式已有的不同。誠實的伺服器每週只使用一把金鑰，因此在下週的金鑰生效之前，本應用程式不會用它領取新的額度。沒有傳送任何請求，也沒有產生任何費用；你已持有的額度仍可正常使用。
space-error-issuer-key-reused = Eidola 伺服器為兩個不同的週提供了同一把付款金鑰，誠實的伺服器絕不會這樣做，因此本應用程式沒有使用它。沒有傳送任何請求，也沒有產生任何費用；你已持有的額度仍可正常使用。
space-error-uncached-retry-unfunded = Eidola 引擎已不再保留此對話先前的上下文，而本應用程式持有的額度不足以完整地重新傳送。沒有再從你的帳戶轉換任何額度。請重試，以新請求的方式傳送。
