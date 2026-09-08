# The onboarding window (src/onboarding/), Traditional Chinese.
#
# `-terms-of-service`, `-privacy-policy`, `onboarding-link-terms-of-service`,
# `onboarding-link-privacy-policy` and `onboarding-account-id-placeholder` are
# deliberately absent: the two published document titles stay English in every
# locale, and the id placeholder is a shape rather than words. They fall back to
# the English source.

onboarding-pause-body =
    ## *請先停一下*

    Eidola *不同於* ChatGPT、Claude 或 Gemini。

onboarding-tool-body =
    ## Eidola 是*你的*工具

    在過去，一款應用程式是裝在光碟裡交到你手上的：

    - 沒有你的參與，它的行為*不會*自行改變。
    - 你的檔案、計畫、使用習慣與想法*只屬於你*，任何第三方都無從得知。
    - 保障這些性質的是技術的**結構**，*而不是*某家公司的承諾。

    Eidola 盡最大可能貼近這種方式，即使是更適合在資料中心執行的工作負載，也從結構上把使用者的自主權做到最大。

onboarding-control-body =
    ## *你的*控制權

    掌控權只在你手中——不在我們手中，也不在營運硬體的業者手中：

    - **只有你能讀取、保存或分析你的互動內容。** 你的資料只在密封的、經硬體證明的飛地中解密，而這些飛地不保留任何內容；Eidola 處理付款的一側與處理請求的一側在密碼學上是分離的。
    - **只有你能更新 Eidola——無論是在你的裝置上還是在伺服器上。** 在你的用戶端驗證過新版本、並且你決定信任它之前，一切都不會改變。

    不要盲目相信我們的說法，請自行驗證。如果你不知道該如何評估我們的程式碼與架構，**請向你已經信任的、最懂技術的人徵詢意見**。

onboarding-responsibility-body =
    ## *你的*責任

    Eidola 這款工具讓*你*無需高度專業的技術能力或昂貴的硬體，也能更輕鬆地執行 AI 模型。沒有人在旁邊替你把關，因此你必須理解以下幾點：

    - 你將要執行的是 AI 模型，其本質是一組機率。在 Eidola 中執行的模型都可以自由下載，用不用 Eidola 都能使用。
    - 即使是最好的模型也會出錯。它們可能非常有用，但確實會犯錯，也可能表現出意料之外的行為。要評估一個大型模型在所有可能情境下的表現，在數學上是不可能的。
    - 模型本身既沒有固有的記憶，也沒有對外產生影響的能力；它們只能評估資料，並執行你提供給它們的操作。Eidola 讓你更容易理解與設定這些權限，但結果——無論好壞——最終由你負責。

onboarding-get-started-body =
    ## 開始使用

    你需要一些額度才能執行模型。

    你的帳戶只是一個隨機識別碼——購買額度是唯一會觸及付款方式的步驟，即使如此，[我們在結構上也無法把你的請求與它關聯起來]({ $unlinkability })。

onboarding-create-account-body =
    ## 建立帳戶

    請閱讀並理解我們的 { -terms-of-service } 與 { -privacy-policy }。

onboarding-new-account-body =
    ## 你的新帳戶

    你的新帳戶已建立：

onboarding-existing-account-body =
    ## 你已有的帳戶

    請輸入你的帳戶資訊：

onboarding-purchase-body =
    ## 新增額度

    選擇一個方案，或直接購買額度。我們使用 Stripe 處理付款。

    訂閱額度在其計費週期內有效；一次性購買的額度有效期為一年。未使用且未過期的額度可應要求退款。

onboarding-back = 返回上一頁

onboarding-cta-pause = 好，我在聽。
onboarding-cta-understood = 我明白了。
onboarding-cta-new-account = 我需要一個新帳戶。
onboarding-cta-existing-account = 我已經有帳戶了。
onboarding-cta-skip-account = 不使用帳戶繼續——僅使用裝置上的模型。

onboarding-consent-agree = 我同意 { -terms-of-service } 與 { -privacy-policy }。
onboarding-terms-loading = 正在檢查目前的文件…
onboarding-terms-retry = 重試

onboarding-link-external = { $label } ↗
onboarding-link-repository = Eidola 程式碼儲存庫
onboarding-document-version = { $name }（版本 { $version }）

onboarding-cta-create = 建立一個新帳戶。
onboarding-cta-create-pending = 正在建立你的匿名帳戶…

onboarding-account-id = 帳戶 ID
onboarding-account-secret = 帳戶金鑰
onboarding-account-secret-placeholder = 帳戶金鑰
onboarding-copy = 複製
onboarding-copy-label = 複製{ $label }
onboarding-new-account-note = 它只用於新增與消耗額度。帳戶金鑰一旦遺失便無法找回，你將需要建立一個新帳戶。
onboarding-cta-saved = 我已經儲存好了。

onboarding-cta-verify = 查詢帳戶餘額。
onboarding-cta-verify-pending = 正在查詢…
onboarding-verify-missing-credentials = 請同時填寫帳戶 ID 與金鑰。
onboarding-verify-unverifiable = 我們無法驗證該帳戶。請檢查 ID 與金鑰，或改為建立一個新帳戶。
onboarding-verified-balance = 該帳戶有效，餘額為 { $credits } 點額度。
onboarding-cta-existing-purchase = 我想購買更多額度。
onboarding-cta-existing-done = 這樣就可以了。

onboarding-purchase-checkout-note = 結帳會在你的瀏覽器中開啟；額度將存入此帳戶。
onboarding-purchase-subscribed-note = 此帳戶已有訂閱——請在 Settings ▸ Account 中管理。你仍然可以在這裡新增一次性額度。
onboarding-purchase-loading = 正在載入方案…
onboarding-purchase-none-topups = 目前沒有可用的一次性儲值。
onboarding-purchase-none-plans = 目前沒有可用的方案。
onboarding-checkout-stale-mint = 準備期間帳戶發生了變更，因此沒有開啟任何頁面。請重試。
onboarding-cta-purchase-later = 我稍後再購買額度。
