# Android Code Vulnerabilities Early Detection (ACVED)

Due to the constant and growing demands of users, as well as their ever-changing needs, Android applications are being rapidly released and updated. However, in the rush to develop these apps, the focus may be more on functionality rather than security and identifying vulnerabilities in the source code. This is partly because there are not enough automated mechanisms available to assist app developers in mitigating these vulnerabilities.

To address this issue, an AI-powered plugin called Android Code Vulnerability Early Detection (ACVED) can be integrated with Android Studio to provide real-time support for mitigating source code vulnerabilities. As you work on a specific source code line, the plugin can provide the vulnerability status for that line.

ACVED has a highly accurate and efficient ensemble learning model running in the backend, which can detect source code vulnerabilities and their CWE categories with a 95% accuracy rate. Additionally, explainable AI techniques are employed to provide source code vulnerability prediction probabilities for each word.

The model is regularly updated with new training data from the LVDAndro dataset, which allows for the detection of novel vulnerabilities using the ACVED plugin.

Use ACVED User Guide for all the necessary instructions related to the ACVED plugin installation and usage.


![ACVED_Plugin_Vul_Code_Presents](https://user-images.githubusercontent.com/102326773/188329694-73ad7acc-2392-409a-ac11-6f40138bdf21.png)


![vul_code_balloon_only](https://user-images.githubusercontent.com/102326773/188329702-0921a281-d701-4e4c-9289-1f5563000e64.png)

![XAI_Predictions_Vul](https://user-images.githubusercontent.com/102326773/188329708-83816d65-49b0-4cb5-ad21-a21653938757.png)
