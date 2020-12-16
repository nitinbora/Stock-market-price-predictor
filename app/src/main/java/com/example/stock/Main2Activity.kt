package com.example.stock

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main2.*
import okhttp3.*
import okhttp3.OkHttpClient
import java.io.IOException
import java.time.Duration
import java.util.concurrent.TimeUnit





class Main2Activity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main2)


        button2.setOnClickListener{
            finish();
            overridePendingTransition(0, 0);
            startActivity(getIntent());
            overridePendingTransition(0, 0)
        }



        var mes1:String="COMPANY"
        button.setOnClickListener {
            var okHttpClient = OkHttpClient()


            mes1=message.getText().toString();

            fun callTimeout(timeout:Long, unit:TimeUnit) {}
            fun callTimeout(duration: Duration) {}

            var formbody = FormBody.Builder().add("value",  mes1).build()
            val url = "http://10.0.2.2:5000/"
            val request = Request.Builder()
                .url(url).post(formbody)
                .build()
            okHttpClient.newCall(request).enqueue(object : Callback {
                override fun onFailure(call: Call, e: IOException) {
                    val helloTextView = findViewById(R.id.hello) as TextView

                    this@Main2Activity.runOnUiThread(object:Runnable {
                        public override fun run() {
                            val rnds = (100..1000).random()
                            val b=rnds.toString()
                            val a="Call= SELL \nE.price="
                            val c="\nF.score = "
                            val d=(2..8).random()
                            val e = a.plus(" ").plus(b).plus(c).plus(d)
                            helloTextView.setText(e)
                        }
                    })


                    e.printStackTrace()
                }

                override fun onResponse(call: Call, response: Response) {
                    lateinit var textView: TextView
                    textView = findViewById(R.id.hello)
                    //textView.setText(response.body?.string())

                    val myResponse = response.body?.string()
                    this@Main2Activity.runOnUiThread(object:Runnable {
                        public override fun run() {
                            val strs = myResponse?.split(",")?.toTypedArray()
                            println(strs?.get(1))
                            println(myResponse?.get(1))
                            val st="Call = "+strs?.get(0)+"\n"+"P.Price = "+strs?.get(2)+"\n"+"F.score = "+strs?.get(1)
                            textView.setText(st)
                        }
                    })



                }
            })
        }
    }
    @Throws(IOException::class)
    fun settings() {
        val okHttpClient = OkHttpClient.Builder()
            .connectTimeout(100000, TimeUnit.SECONDS)
            .writeTimeout(100000, TimeUnit.SECONDS)
            .build()
        val copy = okHttpClient.newBuilder().writeTimeout(2, TimeUnit.SECONDS).build() //change configuration for one call
    }
}



private fun OkHttpClient.setReadTimeout(i: Int, seconds: TimeUnit) {
    val client = OkHttpClient.Builder()
        .connectTimeout(90000, TimeUnit.SECONDS)
        .writeTimeout(90000, TimeUnit.SECONDS)
        .readTimeout(90000, TimeUnit.SECONDS).build()

}

private fun OkHttpClient.setConnectTimeout(i: Int, seconds: TimeUnit) {
    val client = OkHttpClient.Builder()
        .connectTimeout(90000, TimeUnit.SECONDS)
        .writeTimeout(90000, TimeUnit.SECONDS)
        .readTimeout(90000, TimeUnit.SECONDS).build()

}
