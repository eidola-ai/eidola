# The onboarding window (src/onboarding/), Simplified Chinese.
#
# `-terms-of-service`, `-privacy-policy`, `onboarding-link-terms-of-service`,
# `onboarding-link-privacy-policy` and `onboarding-account-id-placeholder` are
# deliberately absent: the two published document titles stay English in every
# locale, and the id placeholder is a shape rather than words. They fall back to
# the English source.

onboarding-pause-body =
    ## *请先停一下*

    Eidola *不同于* ChatGPT、Claude 或 Gemini。

onboarding-tool-body =
    ## Eidola 是*你的*工具

    在过去，一款应用是装在光盘里交到你手上的：

    - 没有你的参与，它的行为*不会*自行改变。
    - 你的文件、计划、使用习惯和想法*只属于你*，任何第三方都无从得知。
    - 保障这些性质的是技术的**结构**，*而不是*某家公司的承诺。

    Eidola 尽最大可能贴近这种方式，即便对于更适合在数据中心运行的工作负载，也从结构上把用户的自主权做到最大。

onboarding-control-body =
    ## *你的*控制权

    掌控权只在你手中——不在我们手中，也不在运行硬件的运营方手中：

    - **只有你能读取、留存或分析你的交互内容。** 你的数据只在密封的、经硬件证明的飞地中解密，而这些飞地不保留任何内容；Eidola 处理付款的一侧与处理请求的一侧在密码学上是分离的。
    - **只有你能更新 Eidola——无论是在你的设备上还是在服务器上。** 在你的客户端验证过新版本、并且你决定信任它之前，一切都不会改变。

    不要盲目相信我们的说法，请自行验证。如果你不知道该如何评估我们的代码和架构，**请向你已经信任的、最懂技术的人征求意见**。

onboarding-responsibility-body =
    ## *你的*责任

    Eidola 这款工具让*你*无需高度专业的技术能力或昂贵的硬件，也能更轻松地运行 AI 模型。没有人在旁边替你把关，因此你必须理解以下几点：

    - 你将要运行的是 AI 模型，其本质是一组概率。在 Eidola 中运行的模型都可以自由下载，用不用 Eidola 都能使用。
    - 即使是最好的模型也会出错。它们可能非常有用，但确实会犯错，也可能表现出意料之外的行为。要评估一个大模型在所有可能情境下的表现，在数学上是不可能的。
    - 模型本身既没有固有的记忆，也没有对外产生影响的能力；它们只能评估数据，并执行你提供给它们的操作。Eidola 让你更容易理解和配置这些权限，但结果——无论好坏——最终由你负责。

onboarding-get-started-body =
    ## 开始使用

    你需要一些额度才能运行模型。

    你的账户只是一个随机标识——购买额度是唯一会触及支付方式的步骤，即便如此，[我们在结构上也无法把你的请求与它关联起来]({ $unlinkability })。

onboarding-create-account-body =
    ## 创建账户

    请阅读并理解我们的 { -terms-of-service } 和 { -privacy-policy }。

onboarding-new-account-body =
    ## 你的新账户

    你的新账户已创建：

onboarding-existing-account-body =
    ## 你已有的账户

    请输入你的账户信息：

onboarding-purchase-body =
    ## 添加额度

    选择一个方案，或直接购买额度。我们使用 Stripe 处理付款。

    订阅额度在其计费周期内有效；一次性购买的额度有效期为一年。未使用且未过期的额度可应要求退款。

onboarding-back = 返回上一页

onboarding-cta-pause = 好，我在听。
onboarding-cta-understood = 我明白了。
onboarding-cta-new-account = 我需要一个新账户。
onboarding-cta-existing-account = 我已经有账户了。
onboarding-cta-skip-account = 不使用账户继续——仅使用设备上的模型。

onboarding-consent-agree = 我同意 { -terms-of-service } 和 { -privacy-policy }。
onboarding-terms-loading = 正在检查当前文件…
onboarding-terms-retry = 重试

onboarding-link-external = { $label } ↗
onboarding-link-repository = Eidola 代码仓库
onboarding-document-version = { $name }（版本 { $version }）

onboarding-cta-create = 创建一个新账户。
onboarding-cta-create-pending = 正在创建你的匿名账户…

onboarding-account-id = 账户 ID
onboarding-account-secret = 账户密钥
onboarding-account-secret-placeholder = 账户密钥
onboarding-copy = 复制
onboarding-copy-label = 复制{ $label }
onboarding-new-account-note = 它只用于添加和消耗额度。账户密钥一旦丢失便无法找回，你将需要创建一个新账户。
onboarding-cta-saved = 我已经保存好了。

onboarding-cta-verify = 查询账户余额。
onboarding-cta-verify-pending = 正在查询…
onboarding-verify-missing-credentials = 请同时填写账户 ID 和密钥。
onboarding-verify-unverifiable = 我们无法验证该账户。请检查 ID 和密钥，或改为创建一个新账户。
onboarding-verified-balance = 该账户有效，余额为 { $credits } 点额度。
onboarding-cta-existing-purchase = 我想购买更多额度。
onboarding-cta-existing-done = 这样就可以了。

onboarding-purchase-checkout-note = 结账会在你的浏览器中打开；额度将充入此账户。
onboarding-purchase-subscribed-note = 此账户已有订阅——请在 Settings ▸ Account 中管理。你仍然可以在这里添加一次性额度。
onboarding-purchase-loading = 正在加载方案…
onboarding-purchase-none-topups = 目前没有可用的一次性充值。
onboarding-purchase-none-plans = 目前没有可用的方案。
onboarding-checkout-stale-mint = 准备期间账户发生了变更，因此没有打开任何页面。请重试。
onboarding-cta-purchase-later = 我稍后再购买额度。
