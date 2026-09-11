<?php
/**
 * Plugin Name: GBTE RAG Search
 * Description: Search box that answers tennis questions grounded in GBTE's video/podcast library.
 * Version: 1.0
 * Author: Ivan Ozerets
 */
function gbte_search_shortcode() {
    return '
        <div id="gbte-search-container">
            <div id="gbte-search-row">
                <input type="text" id="gbte-question" placeholder="Ask a tennis question...">
                <button id="gbte-search-btn">Search</button>
            </div>
            <div id="gbte-results"></div>
        </div>
    ';
}
add_shortcode("gbte_search", "gbte_search_shortcode");

function gbte_enqueue_search_script() {
	wp_enqueue_script("gbte-search-js", plugin_dir_url(__FILE__) . "search.js", array(), "1.7", true);
	wp_enqueue_style("gbte-search-css", plugin_dir_url(__FILE__) . "style.css", array(), "1.7");
}
add_action("wp_enqueue_scripts", "gbte_enqueue_search_script");
