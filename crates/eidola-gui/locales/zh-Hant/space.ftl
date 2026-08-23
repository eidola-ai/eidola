# The conversation window (src/space_view/mod.rs), Traditional Chinese.

space-error-archived = 此對話已封存，無法接收新的回覆。
space-error-not-joined = 這是你的助理之間開啟的對話。你可以閱讀全部內容；要參與則需加入，而此版本尚無法做到。

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
space-error-response-truncated = 模型把全部長度額度都用在思考上，始終沒有開始作答。沒有新增或取代任何回答——請重試，或換一個更簡短的問題。
space-regenerating-elsewhere = 這則回覆正在重新產生。
space-answer-cut-off = 這則回覆達到了長度上限，在中途停下。
