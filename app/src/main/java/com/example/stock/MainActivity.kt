package com.example.stock

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import okhttp3.*
import java.io.IOException

class MainActivity : AppCompatActivity() {

     lateinit var mTextViewResult: TextView
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        mTextViewResult = findViewById(R.id.textview)
        val client = OkHttpClient()



        var formbody = FormBody.Builder().add("name:",  "WIPRO").build()
        val url = "http://10.0.2.2:5000/post"
        val request = Request.Builder()
            .url(url).post(formbody)
            .build()

        client.newCall(request).enqueue(object: Callback {
            override fun onFailure(call: okhttp3.Call, e: IOException) {
                val helloTextView = findViewById(R.id.textview) as TextView
                helloTextView.setText("nitin2")
                e.printStackTrace()
            }
            @Throws(IOException::class)
            override fun onResponse(call: okhttp3.Call, response: Response) {
                if (response.isSuccessful)
                {
                    val myResponse = response.body?.string()
                    this@MainActivity.runOnUiThread(object:Runnable {
                        public override fun run() {
                            mTextViewResult.setText(myResponse)
                        }
                    })
                }
            }
        })
    }





}
